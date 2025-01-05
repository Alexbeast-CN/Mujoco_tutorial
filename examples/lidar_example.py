import mujoco
import numpy as np
import threading
import time
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from mujoco_viewer.mujoco_viewer import MujocoViewer
from utils.sensorUtils import LaserScan, ThreadedSensor
from utils.logUtils import setup_logger

# 设置日志系统
logger_manager = setup_logger(
    enable_file_output=False, context_names=["base", "sim", "sensor"]
)

# 获取logger
base_logger = logger_manager.get_logger("base")
sim_logger = logger_manager.get_logger("sim")
sensor_logger = logger_manager.get_logger("sensor")

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
            
            <!-- 使用replicate创建36个rangefinder和site -->
            <replicate count="36" euler="0 0 3">
                <site name="laser_array" pos="0.02 0 0" zaxis="1 0 0"/>
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
        <!-- 使用line_laser前缀命名传感器 -->
        <rangefinder name="line_laser" site="laser_array"/>
    </sensor>
</mujoco>
"""


class SimulationThread(threading.Thread):
    def __init__(self, model, data):
        super().__init__()
        self.model = model
        self.data = data
        self.running = threading.Event()
        self._lock = threading.Lock()

        # 创建激光扫描器
        self.laser_scan = LaserScan(model, data, frequency=100, logger=sensor_logger)
        self.threaded_scanner = ThreadedSensor(self.laser_scan)

    def get_latest_frame(self):
        """获取最新的扫描帧"""
        return self.threaded_scanner.reading

    def get_state(self):
        """获取当前仿真状态"""
        with self._lock:
            return np.copy(self.data.qpos), np.copy(self.data.qvel)

    def sync_viewer_state(self, data):
        """同步viewer的状态"""
        with self._lock:
            data.qpos[:] = self.data.qpos
            data.qvel[:] = self.data.qvel
            mujoco.mj_forward(self.model, data)

    def run(self):
        sim_logger.info("仿真线程开始运行")
        self.running.set()

        # 启动传感器
        self.threaded_scanner.start()

        while self.running.is_set():
            try:
                with self._lock:
                    # 计算圆周运动的位置
                    t = time.time()
                    radius = 0.3
                    angular_speed = 1.0
                    self.data.qpos[0] = radius * np.cos(t * angular_speed)
                    self.data.qpos[1] = radius * np.sin(t * angular_speed)
                    self.data.qpos[2] = 0.1

                    # 步进仿真
                    mujoco.mj_step(self.model, self.data)

                time.sleep(0.01)  # 100Hz

            except Exception as e:
                sim_logger.error(f"仿真步进出错: {str(e)}")
                break

        # 停止传感器
        self.threaded_scanner.stop()
        sim_logger.info("仿真线程结束运行")

    def stop(self):
        sim_logger.info("正在停止仿真线程...")
        self.running.clear()


def main():
    try:
        base_logger.info("程序启动")

        # 创建模型和数据
        model = mujoco.MjModel.from_xml_string(MODEL_XML)
        data = mujoco.MjData(model)
        base_logger.info("模型加载成功")

        # 启动查看器
        viewer = MujocoViewer(model, data)
        base_logger.info("查看器创建成功")

        def update_plotter(viewer):
            # 获取最新的扫描帧
            frame = sim_thread.get_latest_frame()
            if frame is None:
                return

            # 获取所有点的位置
            positions = frame.positions
            if len(positions) == 0:
                return

            # 提取x和y坐标
            x_coords = positions[:, 0]  # x坐标
            y_coords = positions[:, 1]  # y坐标

            # 找到坐标范围
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()

            # 添加边距
            margin = 0.1
            x_range = max(x_max - x_min, 0.1)  # 避免范围太小
            y_range = max(y_max - y_min, 0.1)
            x_min -= margin * x_range
            x_max += margin * x_range
            y_min -= margin * y_range
            y_max += margin * y_range

            # 清理之前的图形数据
            fig = viewer.figs[0]
            fig.linename[:] = b""
            fig.linepnt[:] = 0
            fig.linedata[:] = 0

            # 设置图形范围
            fig.range[0][0] = x_min  # x轴最小值
            fig.range[0][1] = x_max  # x轴最大值
            fig.range[1][0] = y_min  # y轴最小值
            fig.range[1][1] = y_max  # y轴最大值

            # 添加新的线条
            viewer.add_line_to_fig("scan_points", fig_idx=0)

            # 添加所有点的y坐标
            for y in y_coords:
                viewer.add_data_to_line("scan_points", y, fig_idx=0)

        # 添加回调函数
        viewer.add_post_render_callback(update_plotter)

        # 创建并启动仿真线程
        sim_thread = SimulationThread(model, data)
        sim_thread.start()

        # 主循环(渲染线程)
        while viewer.is_alive:
            viewer.user_scn.ngeom = 0
            viewer.render()
            time.sleep(1 / 60.0)  # 限制渲染帧率

    except KeyboardInterrupt:
        base_logger.info("程序被用户中断")
    except Exception as e:
        base_logger.error(f"程序运行出错: {str(e)}")
    finally:
        # 清理资源
        base_logger.info("正在清理资源...")
        sim_thread.stop()
        sim_thread.join()
        viewer.close()
        base_logger.info("程序正常退出")


if __name__ == "__main__":
    main()
