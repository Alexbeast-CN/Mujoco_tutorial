import mujoco
import mujoco.viewer
from pynput import keyboard
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

inp = InputParameter(6)
out = OutputParameter(6)

# 设置运动学ruckig参数
inp.max_velocity = [1.5] * 6
inp.max_acceleration = [3.0] * 6
inp.max_jerk = [4.0] * 6

# 设置当前位置
inp.current_position = [0.0] * 6
inp.current_velocity = [0.0] * 6
inp.current_acceleration = [0.0] * 6

# 设置目标位置
inp.target_position = [0.0] * 6
inp.target_velocity = [0.0] * 6
inp.target_acceleration = [0.0] * 6

def get_jnt_qpos():
    for i in range(6):
        inp.current_position[i] = data.qpos[jnt_qpos_id[i]]
        
def gen_target_pos():
    pos = []
    for i in range(6):
        pos.append(np.random.uniform(jnt_ranges[i][0], jnt_ranges[i][1]))
    inp.target_position = pos
    print(f"target position: {pos}")    


reach_target = False
def on_press(key):
    try:
        if reach_target and key == keyboard.Key.space:
            get_jnt_qpos()
            gen_target_pos()

    except AttributeError:
        pass

# 创建键盘监听器
listener = keyboard.Listener(on_press=on_press)
listener.start()

model = mujoco.MjModel.from_xml_path("model/universal_robots_ur10e/ur10e.xml")
data = mujoco.MjData(model)

# 初始化ruckig
otg = Ruckig(6, model.opt.timestep)  # DoFs, control cycle
res = Result.Finished

# 更新运动学
mujoco.mj_forward(model, data)

jnt_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
jnt_ranges = []
jnt_id = []
jnt_qpos_id = []
jnt_qvel_id = []

for name in jnt_names:
    jntid= mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    jnt_ranges.append(model.jnt_range[jntid])
    jnt_id.append(jntid)
    jnt_qpos_id.append(model.jnt_qposadr[jntid])
    jnt_qvel_id.append(model.jnt_dofadr[jntid])

print(f"joint id: {jnt_id}")
print(f"joint qpos id: {jnt_qpos_id}")


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if inp.current_position != inp.target_position:
            reach_target = False
            res = Result.Working
            
        if res == Result.Working:
            res = otg.update(inp, out)
            out.pass_to_input(inp)
        
        if res == Result.Finished:
            reach_target = True
        
        for i in range(6):
            data.qpos[jnt_qpos_id[i]] = out.new_position[i]
            data.qvel[jnt_qvel_id[i]] = 0
        mujoco.mj_step(model, data)
        viewer.sync()

listener.stop()