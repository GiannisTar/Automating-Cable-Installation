Setting UR5e pose (visual)
---------------------------

Quick steps to run the pose helper (visual-only):

1) Create and activate a venv:

```bash
python3 -m venv mujoco-venv
source mujoco-venv/bin/activate
```

2) Install the MuJoCo Python package:

```bash
pip install --upgrade pip
pip install mujoco
```

3) Run the helper script:

```bash
cd ~/Desktop/Mujoco
python scripts/set_ur5e_pose.py
```

Notes:
- The script sets qpos directly and does not enable actuators, so it is safe for a visual-only recreation of the scene.
- If the viewer fails to launch check that your Python `mujoco` package version is compatible with your MuJoCo installation.
