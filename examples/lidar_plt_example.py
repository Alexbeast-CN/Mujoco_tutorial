import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.pltUtils import LidarPlotter

MODEL_XML = """
<mujoco model="laser_scanner">
    <asset>
        <!-- 添加棋盘格纹理 -->
        <texture type="2d" name="grid" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" 
                width="512" height="512" mark="edge" markrgb="0.2 0.3 0.4"/>
        <!-- 创建材质 -->
        <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2" texuniform="true"/>
    </asset>

    <option gravity="0 0 0"/>
    
    <worldbody>
        <!-- 四面墙 -->
        <geom name="wall1" type="box" size="0.5 0.01 0.2" pos="0 0.5 0.2"/>  
        <geom name="wall2" type="box" size="0.5 0.01 0.2" pos="0 -0.5 0.2"/>
        <geom name="wall3" type="box" size="0.01 0.5 0.2" pos="0.5 0 0.2"/>
        <geom name="wall4" type="box" size="0.01 0.5 0.2" pos="-0.5 0 0.2"/>
        
        <!-- 修改地板，添加材质 -->
        <geom name="floor" type="plane" size="2 2 .01" material="grid"/>
        
        <!-- 小球和激光雷达 -->
        <body name="sphere" pos="0 0 0.1">
            <freejoint name="ball_joint"/>
            <geom name="ball" type="sphere" size="0.025" rgba="1 0 0 1"/>
            
            <!-- 使用replicate创建36个rangefinder -->
            <replicate count="36" euler="0 0 10">
                <site name="rf" pos="0.02 0 0" zaxis="1 0 0"/>
            </replicate>
            
        </body>
    </worldbody>
    
    <actuator>
        <motor joint="ball_joint" gear="1 0 0 0 0 0"/>
        <motor joint="ball_joint" gear="0 1 0 0 0 0"/>
        <motor joint="ball_joint" gear="0 0 1 0 0 0"/>
        <motor joint="ball_joint" gear="0 0 0 1 0 0"/>
        <motor joint="ball_joint" gear="0 0 0 0 1 0"/>
        <motor joint="ball_joint" gear="0 0 0 0 0 1"/>
    </actuator>
    
    <sensor>
        <rangefinder site="rf"/>
    </sensor>
</mujoco>
"""

# 创建模型和数据
model = mujoco.MjModel.from_xml_string(MODEL_XML)
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
