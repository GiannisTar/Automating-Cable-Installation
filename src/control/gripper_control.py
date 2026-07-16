#!/usr/bin/env python3
"""
Simple helper to open/close the parallel two-finger gripper.
Usage:
  python3 scripts/gripper_control.py open
  python3 scripts/gripper_control.py close
It will step the simulation a short while to move the gripper joints to the requested positions.
"""
import os
import sys
import time
import numpy as np
import mujoco

DOCKER_MODEL_PATH = "/workspace/scenes/start_scene.xml"
LOCAL_MODEL_PATH = "scenes/start_scene.xml"
MODEL_PATH = DOCKER_MODEL_PATH if os.path.exists("/workspace") else LOCAL_MODEL_PATH

# Gripper target slide positions (meters from joint definition):
GRIPPER_OPEN = 0.0   # fingers retracted (wide open)
GRIPPER_CLOSED = 0.04  # fingers moved inwards (closed)

def set_gripper(model, data, left_id, right_id, value):
    data.ctrl[left_id] = value
    data.ctrl[right_id] = value


def tip_contact_detected(model, data):
    # Check contacts for our tip geoms by name
    left_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_left_tip_geom')
    right_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_right_tip_geom')
    left_inner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_left_inner_geom')
    right_inner_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'grip_right_inner_geom')
    if data.ncon <= 0:
        return False
    for i in range(data.ncon):
        con = data.contact[i]
        # con.geom1/geom2 are int indices of geoms involved
        if con.geom1 in (left_geom_id, left_inner_id, right_geom_id, right_inner_id) or con.geom2 in (left_geom_id, left_inner_id, right_geom_id, right_inner_id):
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print('Usage: gripper_control.py open|close')
        return
    cmd = sys.argv[1].lower()
    if cmd not in ('open', 'close'):
        print('Usage: gripper_control.py open|close')
        return

    target = GRIPPER_OPEN if cmd == 'open' else GRIPPER_CLOSED

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # find actuator indices
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_left')
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'gripper_right')
    if left_id < 0 or right_id < 0:
        raise RuntimeError('Gripper actuators not found in model')

    # read current ctrl target (if set) or assume open
    cur = data.ctrl[left_id] if left_id < len(data.ctrl) else 0.0

    # ramp the gripper command over steps to avoid fast impacts
    steps = 80
    for i in range(steps):
        t = (i+1) / steps
        val = cur + (target - cur) * t
        set_gripper(model, data, left_id, right_id, val)
        mujoco.mj_step(model, data)
        # stop if our tip geoms have made contact
        if cmd == 'close' and tip_contact_detected(model, data):
            print('Tip contact detected during close; stopping ramp early')
            break
        time.sleep(model.opt.timestep)

    print(f'Gripper {cmd} command applied (target={target}).')


if __name__ == '__main__':
    main()
