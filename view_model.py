import mujoco
import mujoco.viewer
from pynput import keyboard

run = False


def on_press(key):
    global run
    try:
        if key == keyboard.Key.space:
            run = True
    except AttributeError:
        pass


# 创建键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

model = mujoco.MjModel.from_xml_path("model/universal_robots_ur10e/ur10e.xml")
data = mujoco.MjData(model)
print("Press 'space' to start simulation")

mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if run:
            mujoco.mj_step(model, data)
        viewer.sync()

listener.stop()