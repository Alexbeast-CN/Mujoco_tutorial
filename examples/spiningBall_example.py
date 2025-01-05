import mujoco
import mujoco.viewer
import numpy as np
import os
import sys

# 将项目根目录添加到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from utils.logUtils import setup_logger
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

XML = """
<mujoco>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01"/>
    </default>

    <worldbody>
        <!-- 添加光源 -->
        <light name="top" pos="0 0 3" dir="0 0 -1" directional="true"/>
        <light name="front" pos="3 0 2" dir="-1 0 -0.5"/>
        
        <!-- 地面 -->
        <geom name="ground" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
        
        <!-- 小球 -->
        <body name="ball" pos="0 0 1">
            <joint name="ball_x" type="slide" axis="1 0 0" range="-2 2"/>
            <joint name="ball_y" type="slide" axis="0 1 0" range="-2 2"/>
            <joint name="ball_z" type="slide" axis="0 0 1" range="0.1 2"/>
            <geom name="ball" type="sphere" size="0.05" rgba="1 0 0 1"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="ball_x" name="ax" gear="100"/>
        <motor joint="ball_y" name="ay" gear="100"/>
        <motor joint="ball_z" name="az" gear="100"/>
    </actuator>
</mujoco>
"""

try:
    # 创建模型
    main_logger.info("正在初始化MuJoCo模型...")
    model = mujoco.MjModel.from_xml_string(XML)
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
    time = 0

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
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        main_logger.info("查看器启动成功")

        while viewer.is_running():
            viewer.user_scn.ngeom = 0

            # 计算目标位置
            x = center[0] + radius * np.cos(angular_speed * time)
            y = center[1] + radius * np.sin(angular_speed * time)
            z = center[2]
            target_pos = np.array([x, y, z])

            # 计算控制
            current_pos = data.xpos[ball_id].copy()
            pos_error = target_pos - current_pos

            kp = 0.5
            data.ctrl[0] = kp * pos_error[0]
            data.ctrl[1] = kp * pos_error[1]

            # 仿真前进
            mujoco.mj_step(model, data)
            time += model.opt.timestep

            # 使用rate_limited方法记录仿真状态
            current_pos = data.xpos[ball_id]
            sim_logger.debug_rate_limited(
                f"小球位置: x={current_pos[0]:.3f}, y={current_pos[1]:.3f}, z={current_pos[2]:.3f}",
                interval=0.5,
            )

            # 轨迹相关
            current_pos = data.xpos[ball_id].copy()
            trajectory_drawer.add_point(current_pos)

            trajectory_drawer.draw_trajectory(
                viewer,
                color=[0, 1, 0, 1],
                width=0.002,
            )

            viewer.sync()

except Exception as e:
    main_logger.error(f"程序运行出错: {str(e)}", exc_info=True)
    raise
finally:
    main_logger.info("程序结束")
