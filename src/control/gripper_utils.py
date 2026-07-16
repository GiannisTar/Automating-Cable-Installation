"""Utility functions for the two-finger gripper: force measurement, boolean contact helpers,
and a high-level close-and-hold routine used by controllers.
"""
import time
import numpy as np
import mujoco

# Gripper constants
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.04
GRIPPER_FORCE_TOL = 0.2
PRONG_VEL_TOL = 1e-3
PRONG_SETTLE_MAX = 200

# Module-level gripper state (populated by init_gripper)
gripper_left_act = -1
gripper_right_act = -1
gripper_closed = False
gripper_hold_value = None


def init_gripper(model):
    """Discover gripper actuator indices from the model and store them in module state.
    Safe to call multiple times.
    """
    global gripper_left_act, gripper_right_act
    la = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_left')
    ra = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_right')
    gripper_left_act = int(la) if la >= 0 else -1
    gripper_right_act = int(ra) if ra >= 0 else -1


def apply_hold(data):
    """If the gripper is marked closed, apply the hold value into data.ctrl so
    other controller logic doesn't overwrite the gripper actuators.
    """
    global gripper_closed, gripper_hold_value, gripper_left_act, gripper_right_act
    if not gripper_closed or gripper_hold_value is None:
        return
    if gripper_left_act >= 0:
        data.ctrl[gripper_left_act] = gripper_hold_value
    if gripper_right_act >= 0:
        data.ctrl[gripper_right_act] = gripper_hold_value


def set_hold_value(val):
    """Mark the gripper as closed and remember the actuator value to hold."""
    global gripper_closed, gripper_hold_value
    gripper_closed = True
    gripper_hold_value = float(val)


def _extract_contact_force(con):
    for attr in ("force", "f"):
        if hasattr(con, attr):
            try:
                arr = np.asarray(getattr(con, attr))
                if arr.size >= 3:
                    return arr[:3].astype(np.float64)
            except Exception:
                continue
    return None


def measure_prong_forces(model, data, left_geom_id, right_geom_id):
    left_total = 0.0
    right_total = 0.0
    any_force_data = False
    for ci in range(data.ncon):
        con = data.contact[ci]
        fvec = _extract_contact_force(con)
        if fvec is None:
            continue
        any_force_data = True
        mag = float(np.linalg.norm(fvec))
        if con.geom1 == left_geom_id or con.geom2 == left_geom_id:
            left_total += mag
        if con.geom1 == right_geom_id or con.geom2 == right_geom_id:
            right_total += mag
    if not any_force_data:
        return None, None
    return left_total, right_total


def _boolean_prong_contacts(model, data, left_inner_id, right_inner_id, left_tip_id, right_tip_id):
    left = False
    right = False
    for ci in range(data.ncon):
        con = data.contact[ci]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        if g1 in (left_inner_id, left_tip_id) or g2 in (left_inner_id, left_tip_id):
            left = True
        if g1 in (right_inner_id, right_tip_id) or g2 in (right_inner_id, right_tip_id):
            right = True
    return left, right


def close_and_hold(model, data, gripper_left_act, gripper_right_act, steps_ramp=80):
    """Close the gripper with a ramp, stop early on contact (force-preferred, boolean fallback),
    perform settling, and return the value to hold (float). This function fully steps the sim
    while it runs.

    Returns: hold_value (float)
    """
    # find geom ids
    left_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_left_tip_geom')
    right_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_right_tip_geom')
    left_inner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_left_inner_geom')
    right_inner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_right_inner_geom')

    # find joint ids and qpos/dof addresses
    left_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_left_joint')
    right_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper_right_joint')
    left_qposadr = int(model.jnt_qposadr[left_jid]) if left_jid >= 0 else -1
    right_qposadr = int(model.jnt_qposadr[right_jid]) if right_jid >= 0 else -1
    left_dofadr = int(model.jnt_dofadr[left_jid]) if left_jid >= 0 else -1
    right_dofadr = int(model.jnt_dofadr[right_jid]) if right_jid >= 0 else -1

    qpos_hold = data.qpos.copy()
    cur_left = float(data.ctrl[gripper_left_act])
    cur_right = float(data.ctrl[gripper_right_act])

    for ri in range(steps_ramp):
        t = (ri+1) / steps_ramp
        val = cur_left + (GRIPPER_CLOSED - cur_left) * t
        # reapply held actuator commands for non-gripper actuators
        hold_ctrl = data.ctrl.copy()
        for ai in range(model.nu):
            if ai != gripper_left_act and ai != gripper_right_act:
                data.ctrl[ai] = hold_ctrl[ai]
        # apply ramp
        data.ctrl[gripper_left_act] = val
        data.ctrl[gripper_right_act] = val
        # freeze non-gripper qpos
        for qi in range(model.nq):
            if qi != left_qposadr and qi != right_qposadr:
                data.qpos[qi] = qpos_hold[qi]
        # zero non-gripper qvel
        for vi in range(model.nv):
            if vi != left_dofadr and vi != right_dofadr:
                data.qvel[vi] = 0.0
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        # enforce hold after dynamics
        for qi in range(model.nq):
            if qi != left_qposadr and qi != right_qposadr:
                data.qpos[qi] = qpos_hold[qi]
        for vi in range(model.nv):
            if vi != left_dofadr and vi != right_dofadr:
                data.qvel[vi] = 0.0
        mujoco.mj_forward(model, data)
        # check forces
        left_force, right_force = measure_prong_forces(model, data, left_inner_id, right_inner_id)
        # print some debug intermittently
        if ri % max(1, steps_ramp // 8) == 0:
            print(f"[gripper.ramp] step {ri+1}/{steps_ramp} val={val:.4f} left_force={left_force} right_force={right_force}")
        stop_early = False
        if left_force is not None and right_force is not None:
            if left_force >= GRIPPER_FORCE_TOL and right_force >= GRIPPER_FORCE_TOL:
                stop_early = True
                print(f"Prong forces: left={left_force:.3f}N right={right_force:.3f}N -> stopping ramp")
        else:
            # boolean fallback
            contacting_left, contacting_right = _boolean_prong_contacts(model, data, left_inner_id, right_inner_id, left_tip_id, right_tip_id)
            if contacting_left or contacting_right:
                stop_early = True
                print(f"Tip contact (boolean) during ramp: left={contacting_left} right={contacting_right}")
                # asymmetric handling: retract opposite prong slightly
                if contacting_left and not contacting_right:
                    retract_val = max(GRIPPER_OPEN, val - 0.004)
                    data.ctrl[gripper_right_act] = retract_val
                    print(f"  left contacted first -> retract right to {retract_val:.4f}")
                if contacting_right and not contacting_left:
                    retract_val = max(GRIPPER_OPEN, val - 0.004)
                    data.ctrl[gripper_left_act] = retract_val
                    print(f"  right contacted first -> retract left to {retract_val:.4f}")
        if stop_early:
            break

    # hold value
    hold_value = float(data.ctrl[gripper_left_act])
    # settling
    left_dof = left_dofadr
    right_dof = right_dofadr
    settle_count = 0
    settled = False
    while settle_count < PRONG_SETTLE_MAX:
        mujoco.mj_step(model, data)
        lf, rf = measure_prong_forces(model, data, left_inner_id, right_inner_id)
        v_left = abs(data.qvel[left_dof]) if left_dof >= 0 and left_dof < model.nv else 0.0
        v_right = abs(data.qvel[right_dof]) if right_dof >= 0 and right_dof < model.nv else 0.0
        if lf is not None and rf is not None:
            if lf >= GRIPPER_FORCE_TOL and rf >= GRIPPER_FORCE_TOL and v_left < PRONG_VEL_TOL and v_right < PRONG_VEL_TOL:
                settled = True
                break
        else:
            if v_left < PRONG_VEL_TOL and v_right < PRONG_VEL_TOL:
                settled = True
                break
        if settle_count % max(1, PRONG_SETTLE_MAX // 6) == 0:
            print(f"[gripper.settle] step {settle_count}/{PRONG_SETTLE_MAX} vL={v_left:.6g} vR={v_right:.6g} lf={lf} rf={rf}")
        settle_count += 1

    if settled:
        print(f"Prongs settled after {settle_count} steps (vL={v_left:.4g}, vR={v_right:.4g})")
    else:
        print(f"Prong settle timeout after {PRONG_SETTLE_MAX} steps; proceeding anyway (vL={v_left:.4g}, vR={v_right:.4g})")

    return hold_value
