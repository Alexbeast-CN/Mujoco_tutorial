# mujoco 机械臂线缆仿真

## 1. 关节驱动器 actuator

对于机械臂来说，Mujoco 建模时的常见关节驱动类型有：`general`，`motor`，`position`。

对于非常熟悉 Mujoco 的开发者来说，比如 Mujoco 官方的开发者，他们通常会使用 `general` 来控制关节。比如 mujoco menagerie 中的 [kuka_iiwa_14](https://github.com/google-deepmind/mujoco_menagerie/blob/main/kuka_iiwa_14/iiwa14.xml) 的关节驱动类型就是 `general`。它的好处是你可以使用最底层的 api 来定制关节的动力学性能。以尽可能真实的模拟机械臂的物理特性。

然而，如果你对于 Mujoco 没有那么熟悉，但又希望以力矩作为输入量控制关节，那么我推荐你使用使用 `motor` 来控制关节。它简化了关节执行器的参数配置，让你用默认参数就可以实现机械臂的力控效果。

最后是位置式执行器 `position` 。这是实体机械臂中最常用的控制方式。在 Mujoco 中，我们一般也只需要调整关节的 `ctrlrange` 和 pid 的参数。但，即使是使用了 `position` 来控制关节，Mujoco 的关节控制本质上还是基于动力学仿真的。所以，如果我们机械臂末端负载过大，或者你让机械臂以非常大的速度/加速度运动（即增大惯性力矩），那么位置控制就会失效。

<video width="100%" controls>
  <source src="media/dynamic_illustration.mp4" type="video/mp4">
</video>

如果想要做到完全不考虑动力学的关节位置式控制，则需要删除 `actuator` 直接通过 `data.qpos` 和 `data.qvel` 来直接改变机械臂关节的位置。虽然这个方法 Mujoco 官方不推荐，但是它非常接近目前工业界常用的机械臂运动规划的思路。

我们以 mujoco menagerie 中的 [ur10e](https://github.com/google-deepmind/mujoco_menagerie/blob/main/universal_robots_ur10e/ur10e.xml) 机器人为例，首先我们将 `<actuator>` 标签中的内容全部删除。然后使用 Jerk 层无突变的时间最优运动规划器 [ruckig](https://ruckig.com/) 生成关节位置，然后在每一个仿真步进之前，直接写入 ruckig 计算出的关节位置。

> 拓展阅读：
> 1. [Ruckig arxiv 论文](https://arxiv.org/abs/2105.04830)
> 2. [Ruckig 知乎博客](https://zhuanlan.zhihu.com/p/710371716)

下面是该方法的最小化示例代码。

```python
import mujoco
import mujoco.viewer
from pynput import keyboard
import numpy as np
from ruckig import InputParameter, OutputParameter, Result, Ruckig

# 初始化 Ruckig 运动规划器
inp = InputParameter(6)
out = OutputParameter(6)

# 运动学参数配置
inp.max_velocity = [1.5] * 6
inp.max_acceleration = [3.0] * 6
inp.max_jerk = [4.0] * 6

# 初始化位置和目标
inp.current_position = [0.0] * 6
inp.current_velocity = [0.0] * 6
inp.current_acceleration = [0.0] * 6
inp.target_position = [0.0] * 6
inp.target_velocity = [0.0] * 6
inp.target_acceleration = [0.0] * 6

# 初始化 MuJoCo 环境
model = mujoco.MjModel.from_xml_path("model/universal_robots_ur10e/ur10e.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# 获取关节信息
joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", 
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
joint_ranges = []
joint_qpos_ids = []
joint_qvel_ids = []

for name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    joint_ranges.append(model.jnt_range[joint_id])
    joint_qpos_ids.append(model.jnt_qposadr[joint_id])
    joint_qvel_ids.append(model.jnt_dofadr[joint_id])

# 运动控制相关变量
otg = Ruckig(6, model.opt.timestep)
reach_target = False
ruckig_result = Result.Finished

def update_current_position():
    """更新当前关节位置"""
    for i in range(6):
        inp.current_position[i] = data.qpos[joint_qpos_ids[i]]

def generate_random_target():
    """生成随机目标位置"""
    pos = [np.random.uniform(joint_ranges[i][0], joint_ranges[i][1]) for i in range(6)]
    inp.target_position = pos
    print(f"目标位置: {pos}")

def on_key_press(key):
    """键盘控制回调函数"""
    global reach_target
    try:
        if reach_target and key == keyboard.Key.space:
            update_current_position()
            generate_random_target()
    except AttributeError:
        pass

# 启动键盘监听
listener = keyboard.Listener(on_press=on_key_press)
listener.start()

# 主循环
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # 更新运动状态
        if inp.current_position != inp.target_position:
            reach_target = False
            ruckig_result = Result.Working
            
        if ruckig_result == Result.Working:
            ruckig_result = otg.update(inp, out)
            out.pass_to_input(inp)
        
        if ruckig_result == Result.Finished:
            reach_target = True
        
        # 更新关节位置
        for i in range(6):
            data.qpos[joint_qpos_ids[i]] = out.new_position[i]
            data.qvel[joint_qvel_ids[i]] = 0
            
        mujoco.mj_step(model, data)
        viewer.sync()

listener.stop()
```

<video width="100%" controls>
  <source src="media/positional_control.mp4" type="video/mp4">
</video>