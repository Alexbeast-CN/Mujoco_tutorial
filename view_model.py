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

# model = mujoco.MjModel.from_xml_path("model/universal_robots_ur10e/ur10e.xml")
# model = mujoco.MjModel.from_xml_path("model/kuka_kr20/kuka_kr20.xml")
# model = mujoco.MjModel.from_xml_path("model/KR_20_R1810-2_V00/KR_20_R1810-2_V00.xml")
model = mujoco.MjModel.from_xml_path("model/KR_20_R1810-2_V00/cable_test.xml")
data = mujoco.MjData(model)
print("Press 'space' to start simulation")

mujoco.mj_forward(model, data)
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if run:
            mujoco.mj_step(model, data)
        viewer.sync()

listener.stop()