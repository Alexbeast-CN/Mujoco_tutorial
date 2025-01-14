import random
import time
import mujoco
import mujoco.viewer
from pynput import keyboard
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import manipulation
from utils.logUtils import setup_logger

# 设置日志系统
logger_manager = setup_logger(
    log_dir="logs",
    log_file_prefix="manipulation_movej_example",
    context_names=["main"],
    log_level="INFO",
)
logger = logger_manager.get_logger("main")

# 初始化 MuJoCo 环境
logger.info("正在初始化 MuJoCo 环境...")
# model = mujoco.MjModel.from_xml_path("model/KR_20_R1810-2_V00/KR_20_R1810-2_V00.xml")
model = mujoco.MjModel.from_xml_path("model/kuka_kr20/kuka_kr20.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
logger.info("MuJoCo 环境初始化完成")

# 初始化关节信息
joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
joint_info = manipulation.JointInfo(model, joint_names, 'manipulation/configs/ur10e.yaml')
logger.info(f"已初始化关节: {joint_names}")

# 初始化轨迹规划器
generator = manipulation.MotionGenerator(joint_info, model.opt.timestep)
logger.info("轨迹规划器初始化完成")

run = True
def on_key_press(key):
    global run
    """键盘控制回调函数"""
    try:
        if generator.reach_target and key == keyboard.Key.space:
            run = True
            logger.info("检测到空格键按下,准备更新目标位置")
            
            # 更新当前位置
            current_pos = [data.qpos[i] for i in joint_info.joint_qpos_ids]
            generator.update_current_position(current_pos)
            logger.debug(f"当前位置: {current_pos}")
            
            # 生成新的目标位置
            target_pos = [random.random() if i not in [3,5] else random.uniform(-3.14, 3.14) for i in range(6)]
            generator.set_joint_position(target_pos)
            logger.info(f"已设置新的目标位置: {target_pos}")
    except AttributeError:
        logger.error("键盘事件处理出错", exc_info=True)

# 启动键盘监听
listener = keyboard.Listener(on_press=on_key_press)
listener.start()
logger.info("\033[32m按下空格以生成随机目标关节位置\033[0m")

# 主循环
logger.info("开始主循环...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        if run:
            # 更新轨迹
            new_positions, _ = generator.update()
            
            # 更新关节位置
            for i in range(6):
                data.qpos[joint_info.joint_qpos_ids[i]] = new_positions[i]
                data.qvel[joint_info.joint_qvel_ids[i]] = 0
                data.qacc[joint_info.joint_qvel_ids[i]] = 0
                # data.qacc_warmstart[joint_info.joint_qvel_ids[i]] = 0
            mujoco.mj_step(model, data)
            
        viewer.sync()

listener.stop()
logger.info("程序结束")