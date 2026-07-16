Download the official MuJoCo UR5e model and meshes from the DeepMind Model Zoo:

1. Visit: https://github.com/deepmind/mujoco_menagerie/tree/main/ur5e
2. Download the following files and folders:
   - ur5e.xml
   - meshes/ (entire folder)
3. Place both in this directory: /workspace/models/ur5e/

After downloading, update your demo script to use:

mjcf_path = "/workspace/models/ur5e/ur5e.xml"

You can now run your MuJoCo demo without any URDF conversion or mesh path issues.
