import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.pltUtils import LidarPlotter

# 创建模型和数据
model_path = os.path.join(project_root, "model/lidar_example.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# 创建线程控制标志
viewer_running = True
sim_running = True

# 创建数据锁
data_lock = threading.Lock()

# 创建可视化器
plotter = LidarPlotter(
    num_sensors=36, angle_range=(0, 360), max_range=1.0, update_interval=30
)


# 查看器线程函数
def run_viewer():
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        while viewer_running:
            if not viewer.is_running():
                plotter.close()
                break
            with data_lock:
                viewer.sync()
            time.sleep(0.01)


# 仿真线程函数
def run_simulation():
    while sim_running:
        with data_lock:
            # 计算圆周运动的位置
            t = time.time()
            radius = 0.3
            angular_speed = 1.0
            data.qpos[0] = radius * np.cos(t * angular_speed)
            data.qpos[1] = radius * np.sin(t * angular_speed)
            data.qpos[2] = 0.1

            # 步进仿真
            mujoco.mj_step(model, data)

        # 控制仿真步进频率
        time.sleep(0.01)  # 100Hz


def update_sensor_data():
    """更新传感器数据的回调函数"""
    ranges = []
    with data_lock:
        # 只读取传感器数据,不进行仿真步进
        for i in range(36):
            range_value = data.sensordata[i]
            if range_value < 0:
                range_value = 1.0
            ranges.append(range_value)

    return ranges


def cleanup():
    """清理资源的函数"""
    global viewer_running, sim_running
    viewer_running = False
    sim_running = False
    viewer_thread.join()
    sim_thread.join()


# 启动查看器线程
viewer_thread = threading.Thread(target=run_viewer)
viewer_thread.start()

# 启动仿真线程
sim_thread = threading.Thread(target=run_simulation)
sim_thread.start()

# 启动可视化
plotter.start(update_sensor_data, close_callback=cleanup)
