#!/usr/bin/env python3
"""
Cartesian PID position+orientation controller for UR5e.
- Targets specified as x y z roll pitch yaw (degrees) per target.
- Uses site "attachment_site" on wrist_3_link as the end-effector.
- Maps 6D task error (pos+ori) to joint velocities via damped least-squares inverse of 6xnv Jacobian.
- Integrates joint velocities to produce desired joint positions and sends them to position-style actuators.

Usage examples:
  python3 scripts/ur5e_pid_pose_controller.py 0.8 0.6 0.9 0 0 90
  # two targets:
  python3 scripts/ur5e_pid_pose_controller.py 0.8 0.6 0.9 0 0 90 1.0 0.3 0.9 0 0 0
  # targets in robot-local base frame (use --local)
  python3 scripts/ur5e_pid_pose_controller.py --local 0.5 0 0.2 0 0 0

Notes:
- Orientation input is roll, pitch, yaw in degrees (ZYX intrinsic/extrinsic ordering interpreted as R = Rz(yaw)*Ry(pitch)*Rx(roll)).
- Tune gains (KP_POS, KI_POS, KD_POS, KP_ORI, KI_ORI, KD_ORI) for speed/stability.
"""

import os
import sys
import time
import math
import numpy as np
import mujoco

# Paths
DOCKER_MODEL_PATH = "/workspace/scenes/start_scene.xml"
LOCAL_MODEL_PATH = "scenes/start_scene.xml"
MODEL_PATH = DOCKER_MODEL_PATH if os.path.exists("/workspace") else LOCAL_MODEL_PATH

# Controller gains (tune)
KP_POS = 30.0
KI_POS = 0.0
KD_POS = 12.0
KP_ORI = 15.0
KI_ORI = 0.0
KD_ORI = 6.0
LAMBDA = 0.01

# Limits and criteria
MAX_QDOT = 0.5
POS_TOL = 2e-3  # 2 mm
ORI_TOL = 0.02  # ~1 degree
HOLD_STEPS = 50
MAX_STEPS = 10000

SITE_NAME = "attachment_site"
BASE_BODY_NAME = "base"


def deg2rad(x):
    return (x / 180.0) * math.pi


def rpy_to_quat(roll, pitch, yaw):
    # roll-pitch-yaw to quaternion w,x,y,z for intrinsic rotations
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = cy * sp * cr + sy * cp * sr
    z = sy * cp * cr - cy * sp * sr
    return np.array([w, x, y, z], dtype=np.float64)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a, b):
    # Hamilton product
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_to_axis_angle(q):
    # q assumed normalized [w,x,y,z]
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(0.0, 1.0 - w*w))
    if s < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = q[1:4] / s
    return axis * angle


def damped_pinv(J, lam):
    JJt = J @ J.T
    reg = (lam*lam) * np.eye(J.shape[0])
    return J.T @ np.linalg.inv(JJt + reg)


def parse_poses_from_argv(argv):
    # Expects triples of 6 numbers: x y z roll pitch yaw (degrees)
    if len(argv) <= 1:
        return None
    try:
        vals = [float(a) for a in argv[1:]]
    except ValueError:
        print("Non-numeric pose value provided; falling back to default.")
        return None
    if len(vals) % 6 != 0:
        print("Provide poses as multiples of six numbers: x y z roll pitch yaw ...")
        return None
    poses = []
    for i in range(0, len(vals), 6):
        x, y, z, rr, rp, ry = vals[i:i+6]
        q = rpy_to_quat(deg2rad(rr), deg2rad(rp), deg2rad(ry))
        poses.append((np.array([x,y,z], dtype=np.float64), q))
    return poses


def pop_local_flag(argv):
    if "--local" in argv:
        argv = [a for a in argv if a != "--local"]
        return argv, True
    return argv, False


def get_site_state(model, data, site_id):
    pos = data.site_xpos[site_id].copy()
    # orientation: prefer site_xquat if present, else build from site_xmat
    if hasattr(data, 'site_xquat') and len(data.site_xquat) > site_id:
        quat = data.site_xquat[site_id].copy()
    else:
        # fallback: use site_xmat (9) -> reshape
        mat = data.site_xmat[site_id].copy().reshape(3,3)
        # convert rot mat to quat
        # from https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        m = mat
        tr = m[0,0] + m[1,1] + m[2,2]
        if tr > 0:
            S = math.sqrt(tr+1.0) * 2.0
            w = 0.25 * S
            x = (m[2,1] - m[1,2]) / S
            y = (m[0,2] - m[2,0]) / S
            z = (m[1,0] - m[0,1]) / S
        elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
            w = (m[2,1] - m[1,2]) / S
            x = 0.25 * S
            y = (m[0,1] + m[1,0]) / S
            z = (m[0,2] + m[2,0]) / S
        elif m[1,1] > m[2,2]:
            S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
            w = (m[0,2] - m[2,0]) / S
            x = (m[0,1] + m[1,0]) / S
            y = 0.25 * S
            z = (m[1,2] + m[2,1]) / S
        else:
            S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
            w = (m[1,0] - m[0,1]) / S
            x = (m[0,2] + m[2,0]) / S
            y = (m[1,2] + m[2,1]) / S
            z = 0.25 * S
        quat = np.array([w,x,y,z], dtype=np.float64)
    return pos, quat


def main():
    argv, local_mode = pop_local_flag(sys.argv)
    poses = parse_poses_from_argv(argv) or [ (np.array([0.8,0.6,0.9],dtype=np.float64), rpy_to_quat(0,0,0)) ]
    print("Poses:")
    for i,(p,q) in enumerate(poses,1):
        print(f"  {i}: pos={p}, quat={q}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # initialize
    q_init = np.array([0, -1.5708, 1.5708, 0, 1.5708, 0])
    data.qpos[: len(q_init)] = q_init
    mujoco.mj_forward(model, data)

    # if local mode, transform poses
    if local_mode:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
        if base_id < 0:
            raise RuntimeError("Base body not found for local transform")
        # robustly get base pose: prefer runtime data, fall back to model defaults
        try:
            base_pos = data.body_xpos[base_id].copy()
            base_quat = data.body_xquat[base_id].copy()
        except Exception:
            # fallback to model defaults (static from XML)
            base_pos = model.body_pos[base_id].copy()
            base_quat = model.body_quat[base_id].copy()
        # build rotation matrix
        w,x,y,z = base_quat
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        world_poses = []
        for p,q in poses:
            world_p = base_pos + R.dot(p)
            # world orientation = base_quat * local_quat
            world_q = quat_mul(base_quat, q)
            world_poses.append((world_p, world_q))
        poses = world_poses
        print("Transformed targets to world frame:")
        for i,(p,q) in enumerate(poses,1):
            print(f"  {i}: pos={p}, quat={q}")

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SITE_NAME)
    if site_id < 0:
        raise RuntimeError(f"Site '{SITE_NAME}' not found in model")

    nv = model.nv
    nu = model.nu

    # controller state
    pos_int = np.zeros(3, dtype=np.float64)
    ori_int = np.zeros(3, dtype=np.float64)
    q_des = data.qpos.copy()

    dt = model.opt.timestep
    target_index = 0
    within = 0

    traj = []

    for step in range(MAX_STEPS):
        # current state
        x, q_cur = get_site_state(model, data, site_id)
        # get site velocity (linear + angular)
        v6 = np.zeros(6, dtype=np.float64)
        try:
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, 1, v6)
            v_lin = v6[:3].copy()
            v_ang = v6[3:6].copy()
        except Exception:
            v_lin = np.zeros(3, dtype=np.float64)
            v_ang = np.zeros(3, dtype=np.float64)

        # target
        p_d, q_d = poses[target_index]

        # position error
        e_pos = p_d - x
        pos_int += e_pos * dt
        e_pos_dot = -v_lin
        F_pos = KP_POS * e_pos + KI_POS * pos_int + KD_POS * e_pos_dot

        # orientation error via quaternion: q_err = q_d * q_cur^-1
        qcur_conj = quat_conj(q_cur)
        q_err = quat_mul(q_d, qcur_conj)
        e_ori = quat_to_axis_angle(q_err)  # 3-vector = axis*angle
        ori_int += e_ori * dt
        e_ori_dot = -v_ang
        F_ori = KP_ORI * e_ori + KI_ORI * ori_int + KD_ORI * e_ori_dot

        # full wrench
        wrench = np.zeros(6, dtype=np.float64)
        wrench[:3] = F_pos
        wrench[3:6] = F_ori

        # Jacobian 6 x nv
        jacp = np.zeros((3, nv), dtype=np.float64)
        jacr = np.zeros((3, nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])

        J_pinv = damped_pinv(J, LAMBDA)
        qdot_cmd = J_pinv @ wrench
        # clamp
        qdot_cmd = np.clip(qdot_cmd, -MAX_QDOT, MAX_QDOT)

        q_des[:nv] += qdot_cmd[:nv] * dt
        ctrl = q_des[:nu].copy()
        # clamp ctrl to actuator ranges if present
        if model.actuator_ctrlrange is not None:
            lo = model.actuator_ctrlrange[:,0]
            hi = model.actuator_ctrlrange[:,1]
            ctrl = np.clip(ctrl, lo, hi)
        data.ctrl[:nu] = ctrl

        mujoco.mj_step(model, data)
        traj.append(data.qpos.copy())

        pos_err_norm = np.linalg.norm(e_pos)
        ori_err_norm = np.linalg.norm(e_ori)

        if pos_err_norm < POS_TOL and ori_err_norm < ORI_TOL:
            within += 1
        else:
            within = 0

        if within >= HOLD_STEPS:
            print(f"Reached target {target_index+1} step {step}: pos={x}, ori_err={ori_err_norm}")
            # hold: zero velocities and command current qpos
            q_des[:nv] = data.qpos.copy()
            for _ in range(60):
                data.qvel[:nv] = 0.0
                data.ctrl[:nu] = q_des[:nu].copy()
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
            target_index += 1
            within = 0
            if target_index >= len(poses):
                print("All targets reached")
                break

        if step % 200 == 0:
            print(f"Step {step}, target {target_index+1}, pos_err={pos_err_norm:.4f}, ori_err={ori_err_norm:.4f}")

    # save
    out_dir = "/workspace/output" if os.path.exists("/workspace") else "output"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "ur5e_pid_pose_traj.npz"), trajectory=np.array(traj))
    print(f"Saved trajectory to {os.path.join(out_dir, 'ur5e_pid_pose_traj.npz')}")


if __name__ == '__main__':
    main()
