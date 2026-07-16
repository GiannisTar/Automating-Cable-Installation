import os
import sys
try:
    import mujoco
except Exception as e:
    print('Error: could not import mujoco. Install the Python package first (see script header).')
    raise

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT, 'models', 'empty_scene.xml')

if not os.path.exists(MODEL_PATH):
    print('Model not found:', MODEL_PATH)
    sys.exit(1)

# Approximate joint angles (radians) chosen to match photo-like pose.
pose = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -1.0,
    'elbow_joint': 1.5,
    'wrist_1_joint': -1.0,
    'wrist_2_joint': -1.2,
    'wrist_3_joint': 0.0,
    # gripper slides (keep nearly closed visually)
    'gripper_left_joint': 0.01,
    'gripper_right_joint': 0.01,
}


def apply_pose_and_render(model_path, pose, out_image):
    # Support both old (MjSim) and new (MjModel+MjData) APIs.
    # Load model
    model = mujoco.MjModel.from_xml_path(model_path)

    # create data
    try:
        data = mujoco.MjData(model)
    except Exception:
        # older API path (MjSim)
        try:
            sim = mujoco.MjSim(model)
            for name, val in pose.items():
                try:
                    jid = sim.model.joint_name2id(name)
                    qaddr = int(sim.model.jnt_qposadr[jid])
                    sim.data.qpos[qaddr] = float(val)
                except Exception:
                    print('Warning: could not set joint', name)
            sim.forward()
            # try viewer if available
            try:
                from mujoco.viewer import launch
                launch(sim)
                return True
            except Exception:
                print('No Python viewer available for this mujoco version; pose applied to sim.')
                return True
        except Exception as e:
            print('Failed to create simulation:', e)
            return False

    # new API: set qpos via name lookup
    for name, val in pose.items():
        try:
            # get joint id using name2id
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                print('Warning: joint not found in model:', name)
                continue
            addr = int(model.jnt_qposadr[jid])
            data.qpos[addr] = float(val)
        except Exception as e:
            print('Warning setting', name, '->', e)

    # forward kinematics
    mujoco.mj_forward(model, data)

    # render offscreen using Renderer and save an image
    try:
        renderer = mujoco.Renderer(model, height=800, width=1200)
        renderer.update_scene(data)
        img = renderer.render()
        try:
            import imageio
            imageio.imwrite(out_image, img)
            print('Wrote image to', out_image)
        except Exception as e:
            print('Rendered image available as numpy array. To write it to file, install imageio:', e)
    except Exception as e:
        print('Rendering failed:', e)
        return False

    return True


out_path = os.path.join(ROOT, 'ur5e_pose.png')
ok = apply_pose_and_render(MODEL_PATH, pose, out_path)
if ok:
    print('Done. Open', out_path)
else:
    print('Failed to apply pose and render. You can still open the model in simulate and adjust manually.')
    
