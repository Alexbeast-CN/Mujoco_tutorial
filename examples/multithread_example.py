import mujoco
import numpy as np
import traceback
import threading
import time
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from utils.logUtils import setup_logger
from mujoco_viewer.mujoco_viewer import MujocoViewer
from utils.vizUtils import TrajectoryDrawer

# 设置日志系统
logger_manager = setup_logger(
    log_dir="logs",
    log_file_prefix="spinning_ball",
    context_names=["main", "simulation", "trajectory"],
    log_level="INFO",
)

# 获取不同上下文的logger
main_logger = logger_manager.get_logger("main")
sim_logger = logger_manager.get_logger("simulation")
traj_logger = logger_manager.get_logger("trajectory")


class SimulationThread(threading.Thread):
    def __init__(self, model, data, center, radius, angular_speed, ball_id):
        super().__init__()
        self.model = model
        self.data = data
        self.center = center
        self.radius = radius
        self.angular_speed = angular_speed
        self.ball_id = ball_id
        self.sim_time = 0
        self.running = threading.Event()

    def run(self):
        sim_logger.info("仿真线程开始运行")
        self.running.set()

        while self.running.is_set():
            try:
                # 计算目标位置
                x = self.center[0] + self.radius * np.cos(
                    self.angular_speed * self.sim_time
                )
                y = self.center[1] + self.radius * np.sin(
                    self.angular_speed * self.sim_time
                )
                z = self.center[2]
                target_pos = np.array([x, y, z])

                # 计算控制
                current_pos = self.data.xpos[self.ball_id].copy()
                pos_error = target_pos - current_pos

                kp = 0.5
                self.data.ctrl[0] = kp * pos_error[0]
                self.data.ctrl[1] = kp * pos_error[1]

                # 仿真前进
                mujoco.mj_step(self.model, self.data)
                self.sim_time += self.model.opt.timestep

                # 记录仿真状态
                self._log_simulation_state()
                
                # 控制仿真频率
                time.sleep(self.model.opt.timestep)

            except Exception as e:
                sim_logger.error(
                    f"仿真线程发生错误: {str(e)}\n"
                )
                break

        sim_logger.info("仿真线程结束运行")

    def stop(self):
        sim_logger.info("正在停止仿真线程...")
        self.running.clear()

    def _log_simulation_state(self):
        current_pos = self.data.xpos[self.ball_id]
        sim_logger.debug_rate_limited(
            f"小球位置: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}, time={self.sim_time:.3f}",
            interval=0.5,
        )


try:
    # 创建模型
    main_logger.info("正在初始化MuJoCo模型...")
    model_path = os.path.join(project_root, "model/1.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    main_logger.info("模型初始化成功")

    # 初始化轨迹绘制器
    trajectory_drawer = TrajectoryDrawer(max_segments=5000, min_distance=0.02)
    traj_logger.info(f"轨迹绘制器初始化: max_segments={5000}, min_distance={0.02}")

    # 获取小球的 body id
    ball_id = model.body("ball").id

    # 圆周运动参数
    radius = 0.5
    center = np.array([0, 0, 1])
    angular_speed = 2
    sim_time = 0

    sim_logger.info(
        f"运动参数设置: radius={radius}, center={center}, angular_speed={angular_speed}"
    )

    # 设置初始位置
    initial_x = center[0] + radius * np.cos(0)
    initial_y = center[1] + radius * np.sin(0)
    initial_z = center[2]

    data.qpos[0] = initial_x
    data.qpos[1] = initial_y
    data.qpos[2] = initial_z

    sim_logger.info(
        f"初始位置设置: x={initial_x:.3f}, y={initial_y:.3f}, z={initial_z:.3f}"
    )

    # 启动查看器
    main_logger.info("启动MuJoCo查看器...")
    viewer = MujocoViewer(model, data)

    # 添加轨迹绘制回调
    def draw_trajectory_callback(viewer):
        trajectory_drawer.draw_trajectory(viewer, color=[0, 1, 0, 1], width=0.02)

    viewer.add_post_render_callback(draw_trajectory_callback)
    main_logger.info("查看器启动成功")

    # 添加图表线条
    viewer.add_line_to_fig("x_position", fig_idx=0)
    main_logger.info("添加了x位置图表线条")

    # 创建并启动仿真线程
    sim_thread = SimulationThread(model, data, center, radius, angular_speed, ball_id)
    sim_thread.start()
    main_logger.info("仿真线程已启动")

    # 主循环，用于更新数据和渲染
    while viewer.is_alive:
        viewer.user_scn.ngeom = 0

        current_pos = data.xpos[ball_id].copy()
        # 更新图表数据
        viewer.add_data_to_line("x_position", current_pos[0], fig_idx=0)
        # 添加轨迹点
        trajectory_drawer.add_point(current_pos)

        viewer.render()
        time.sleep(1 / 60.0)  # 限制渲染帧率

    # 停止仿真线程
    sim_thread.stop()
    sim_thread.join()
    viewer.close()

except Exception as e:
    error_msg = f"程序运行出错: {str(e)}"
    main_logger.error(error_msg)
    if "sim_thread" in locals():
        sim_thread.stop()
        sim_thread.join()
    raise
finally:
    main_logger.info("程序结束")
