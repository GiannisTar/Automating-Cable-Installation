"""
Simple visualizer that plays the PID trajectory with baked-in model and trajectory paths.
Run:
  python3 scripts/ur5e_visualize.py

This intentionally uses fixed paths so no CLI flags are needed.
"""
import os
import time
import numpy as np
import mujoco
from mujoco_viewer import MujocoViewer

MODEL_PATH = "scenes/start_scene.xml"
# TRAJ_PATH = "output/ur5e_pid_pose_traj.npz"
TRAJ_PATH = "output/ur5e_pos_ik_traj.npz"
PLAY_RATE_HZ = 150.0

if not os.path.exists(TRAJ_PATH):
    raise SystemExit(f"Trajectory file not found: {TRAJ_PATH}. Run a controller to generate it first.")

traj = np.load(TRAJ_PATH)["trajectory"]

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
viewer = MujocoViewer(model, data)

# Use model.nq (qpos length) when preparing qpos arrays for playback. Trajectory frames
# may contain only the robot's joint values (nq_robot) while the compiled scene model
# can have extra qpos entries (e.g. free bodies). Pad or truncate accordingly.
nq = model.nq
dt = 1.0 / PLAY_RATE_HZ
for frame in traj:
    # frame may be shorter (e.g. 12) while model.nq may be larger (e.g. 13)
    if frame.shape[0] < nq:
        q = np.zeros(nq)
        q[:frame.shape[0]] = frame
    else:
        q = frame[:nq]
    # final safety: ensure q has correct length
    if q.shape[0] != data.qpos.shape[0]:
        print(f"Warning: q size {q.shape[0]} != data.qpos size {data.qpos.shape[0]}, resizing")
        q = np.resize(q, data.qpos.shape[0])
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    viewer.render()
    time.sleep(dt)

print("Playback finished. Close viewer to exit.")
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        break
