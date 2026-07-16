#!/usr/bin/env python3
"""
Pure position IK controller (iterative damped least-squares) for UR5e.
- Targets specified as x y z roll pitch yaw (degrees) per target.
- Uses site "attachment_site" on wrist_3_link as the end-effector.
- Performs a small number of inner IK iterations each control step to solve for joint positions
  that reduce the pose error, then commands those joint positions to position-style actuators.

Usage examples:
  python3 scripts/ur5e_position_ik_controller.py
  python3 scripts/ur5e_position_ik_controller.py 0.8 0.6 0.9 0 0 90
  python3 scripts/ur5e_position_ik_controller.py --local 0.5 0 0.2 0 0 0

This controller is "pure position" in the sense that it computes joint-position targets
via (damped) IK and writes them directly to actuators each step (no velocity integration loop).
"""

import os
import sys
import math
import time
import numpy as np
import mujoco

# Paths
DOCKER_MODEL_PATH = "/workspace/scenes/start_scene.xml"
LOCAL_MODEL_PATH = "scenes/start_scene.xml"
MODEL_PATH = DOCKER_MODEL_PATH if os.path.exists("/workspace") else LOCAL_MODEL_PATH

# IK & controller params (tune)
# Increased inner iterations and slightly larger per-iteration step to
# improve convergence to the exact target pose.
IK_ITERS = 12           # inner IK iterations per control step
LAMBDA = 0.015          # damping for damped-pinv (smaller -> more aggressive)
STEP_ALPHA = 1.0        # scale factor for delta-q produced by IK
MAX_DELTA_Q = 0.18      # rad per inner iteration max

# Convergence criteria
# Relaxed tolerances so the controller doesn't demand impractically small errors
# Position tolerance: 1 cm, Orientation tolerance: ~0.05 rad (~2.8 deg)
POS_TOL = 1.5e-2        # 1.5 cm
ORI_TOL = 0.05
HOLD_STEPS = 12         # fewer steps required within tolerance before advancing
MAX_STEPS = 8000


SITE_NAME = "attachment_site"
BASE_BODY_NAME = "base"

# Default sequence (x, y, z, roll, pitch, yaw) in degrees used when no CLI poses are given.
# Edit this list to change the built-in demo sequence.
DEFAULT_POSES_DEG = [
    # Absolute Positions, using others due to some position errors
    # [0.65, 0.6, 1.0, 180, 0, 90],
    # [0.65, 0.6, 0.875, 180, 0, 90],
    # [0.65, 0.6, 1.0, 180, 0, 90],
    # [0.25, 0.6, 0.875, 180, 0, 90],
    # [0.2, 0.6, 0.875, 180, 0, 90],
    # [0.25, 0.6, 0.875, 180, 0, 90],
    # [0.65, 0.6, 1.0, 180, 0, 90],
    # [0.65, 0.6, 0.875, 180, 0, 90],
    # [0.65, 0.6, 1.0, 180, 0, 90]

    [0.65, 0.6, 1.0, 180, 0, 90],
    [0.65, 0.6, 0.87, 180, 0, 90],
    [0.65, 0.6, 1.0, 180, 0, 90],
    [0.25, 0.6, 0.9, 180, 0, 90],
    [0.175, 0.6, 0.9, 180, 0, 90],
    [0.25, 0.6, 0.9, 180, 0, 90],
    [0.65, 0.6, 1.0, 180, 0, 90],
    [0.65, 0.6, 0.875, 180, 0, 90],
    [0.65, 0.6, 1.0, 180, 0, 90]
]


def deg2rad(x):
    return (x / 180.0) * math.pi


def rpy_to_quat(roll, pitch, yaw):
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
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)


def quat_to_axis_angle(q):
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
    if hasattr(data, 'site_xquat') and len(data.site_xquat) > site_id:
        quat = data.site_xquat[site_id].copy()
    else:
        mat = data.site_xmat[site_id].copy().reshape(3,3)
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
    poses = parse_poses_from_argv(argv)
    if poses is None:
        # build poses from the built-in DEFAULT_POSES_DEG
        poses = []
        for pdeg in DEFAULT_POSES_DEG:
            pos = np.array(pdeg[0:3], dtype=np.float64)
            quat = rpy_to_quat(deg2rad(pdeg[3]), deg2rad(pdeg[4]), deg2rad(pdeg[5]))
            poses.append((pos, quat))
        print("No CLI poses provided — using built-in sequence:")
    else:
        print("CLI poses:")
    for i,(p,q) in enumerate(poses,1):
        print(f"  {i}: pos={p}, quat={q}")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # initialize robot in a reasonable home
    q_init = np.array([0, -1.5708, 1.5708, 0, 1.5708, 0])
    data.qpos[: len(q_init)] = q_init
    mujoco.mj_forward(model, data)

    # handle local-frame targets
    if local_mode:
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY_NAME)
        if base_id < 0:
            raise RuntimeError("Base body not found for local transform")
        try:
            base_pos = data.body_xpos[base_id].copy()
            base_quat = data.body_xquat[base_id].copy()
        except Exception:
            base_pos = model.body_pos[base_id].copy()
            base_quat = model.body_quat[base_id].copy()
        w,x,y,z = base_quat
        R = np.array([
            [1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=np.float64)
        world_poses = []
        for p,q in poses:
            world_p = base_pos + R.dot(p)
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

    # (gripper steps removed) -- controller only performs IK waypoints now

    dt = model.opt.timestep
    target_index = 0
    within = 0
    prev_target_index = -1

    traj = []

    # temporary data used for inner IK evaluation
    data_temp = mujoco.MjData(model)

    for step in range(MAX_STEPS):
        # announce new target when we switch
        if target_index != prev_target_index:
            p_d_tmp, q_d_tmp = poses[target_index]
            print(f"Starting target {target_index+1}/{len(poses)}: pos={p_d_tmp}")
            prev_target_index = target_index
        # per-iteration marker to note if we performed a gripper ramp/settle
        did_ramp = False

        # current end-effector pose (before solving IK for this step)
        x_cur, q_cur = get_site_state(model, data, site_id)
        p_d, q_d = poses[target_index]

        # inner IK: try to find q_temp that reduces full pose error
        q_temp = data.qpos.copy()
        for k in range(IK_ITERS):
            # evaluate FK/Jac at q_temp using data_temp
            data_temp.qpos[:] = q_temp
            mujoco.mj_forward(model, data_temp)
            # recompute current pose at q_temp
            x_tmp = data_temp.site_xpos[site_id].copy()
            if hasattr(data_temp, 'site_xquat') and len(data_temp.site_xquat) > site_id:
                q_tmp = data_temp.site_xquat[site_id].copy()
            else:
                mat = data_temp.site_xmat[site_id].copy().reshape(3,3)
                # convert mat to quat (same routine as elsewhere)
                m = mat
                tr = m[0,0] + m[1,1] + m[2,2]
                if tr > 0:
                    S = math.sqrt(tr+1.0) * 2.0
                    w = 0.25 * S
                    xq = (m[2,1] - m[1,2]) / S
                    yq = (m[0,2] - m[2,0]) / S
                    zq = (m[1,0] - m[0,1]) / S
                elif (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
                    S = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2.0
                    w = (m[2,1] - m[1,2]) / S
                    xq = 0.25 * S
                    yq = (m[0,1] + m[1,0]) / S
                    zq = (m[0,2] + m[2,0]) / S
                elif m[1,1] > m[2,2]:
                    S = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2.0
                    w = (m[0,2] - m[2,0]) / S
                    xq = (m[0,1] + m[1,0]) / S
                    yq = 0.25 * S
                    zq = (m[1,2] + m[2,1]) / S
                else:
                    S = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2.0
                    w = (m[1,0] - m[0,1]) / S
                    xq = (m[0,2] + m[2,0]) / S
                    yq = (m[1,2] + m[2,1]) / S
                    zq = 0.25 * S
                q_tmp = np.array([w, xq, yq, zq], dtype=np.float64)

            # compute error at q_temp
            e_pos_tmp = p_d - x_tmp
            qtmp_conj = quat_conj(q_tmp)
            q_err_tmp = quat_mul(q_d, qtmp_conj)
            e_ori_tmp = quat_to_axis_angle(q_err_tmp)
            err6 = np.hstack([e_pos_tmp, e_ori_tmp])

            # Jacobian at q_temp
            jacp = np.zeros((3, nv), dtype=np.float64)
            jacr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacSite(model, data_temp, jacp, jacr, site_id)
            J = np.vstack([jacp, jacr])

            J_pinv = damped_pinv(J, LAMBDA)
            delta_q = J_pinv @ err6
            # scale and clamp
            delta_q = STEP_ALPHA * delta_q
            delta_q = np.clip(delta_q, -MAX_DELTA_Q, MAX_DELTA_Q)
            q_temp[:nv] += delta_q[:nv]

        # command the IK result as joint position targets
        q_des = q_temp
        ctrl = q_des[:nu].copy()
        if model.actuator_ctrlrange is not None:
            lo = model.actuator_ctrlrange[:,0]
            hi = model.actuator_ctrlrange[:,1]
            ctrl = np.clip(ctrl, lo, hi)
        data.ctrl[:nu] = ctrl

    # ...gripper logic removed; the controller only commands IK targets.

        # step simulation
        # After stepping, re-evaluate the true end-effector pose/error so that
        # the "within" counter uses the post-step pose (what the robot actually reached).
        mujoco.mj_step(model, data)
        traj.append(data.qpos.copy())

        # re-evaluate current pose/error (post-step)
        x_cur, q_cur = get_site_state(model, data, site_id)
        e_pos = p_d - x_cur
        qcur_conj = quat_conj(q_cur)
        q_err = quat_mul(q_d, qcur_conj)
        e_ori = quat_to_axis_angle(q_err)

        pos_err_norm = np.linalg.norm(e_pos)
        ori_err_norm = np.linalg.norm(e_ori)

        if pos_err_norm < POS_TOL and ori_err_norm < ORI_TOL:
            within += 1
        else:
            within = 0
        if within >= HOLD_STEPS:
            print(f"Reached target {target_index+1} step {step}: pos={x_cur}, ori_err={ori_err_norm}")
            # Advance to the next waypoint (gripper behavior removed).
            target_index += 1
            within = 0
            if target_index >= len(poses):
                print("All targets reached")
                break

        if step % 200 == 0:
            print(f"Step {step}, target {target_index+1}, pos_err={pos_err_norm:.4f}, ori_err={ori_err_norm:.4f}")

    out_dir = "/workspace/output" if os.path.exists("/workspace") else "output"
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "ur5e_pos_ik_traj.npz"), trajectory=np.array(traj))
    print(f"Saved trajectory to {os.path.join(out_dir, 'ur5e_pos_ik_traj.npz')}")


if __name__ == '__main__':
    main()
