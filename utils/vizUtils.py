import mujoco
import numpy as np
from utils.logUtils import setup_logger
from utils.sensorUtils import SensorNameGenerator
from utils.geoUtils import RDPSimplifier
import threading


class TrajectoryDrawer:
    def __init__(
        self,
        max_segments=500,
        min_distance=0.01,
        rdp_epsilon=0.01,
        window_size=100,
        auto_simplify=True,
        min_points_to_simplify=50,
    ):
        """
        轨迹绘制器的初始化。

        参数：
        - max_segments: 最大显示的轨迹线段数
        - min_distance: 相邻轨迹点之间的最小距离
        - rdp_epsilon: RDP算法的简化阈值
        - window_size: 简化处理的滑动窗口大小
        - auto_simplify: 是否启用自动简化
        - min_points_to_simplify: 触发简化的最小点数
        """
        self.max_segments = max_segments
        self.min_distance = min_distance
        self.rdp_epsilon = rdp_epsilon
        self.window_size = window_size
        self.auto_simplify = auto_simplify
        self.min_points_to_simplify = min_points_to_simplify

        self.positions = []  # 存储有效的轨迹点
        self.current_point = None  # 当前点
        self.accumulated_distance = 0.0  # 累积距离
        self.last_simplified_index = 0  # 上次简化处理到的位置
        self.positions_lock = threading.Lock()  # 添加锁

    def add_point(self, point):
        """
        添加新的轨迹点。

        参数：
        - point: 新的位置点
        """
        point = np.array(point)

        with self.positions_lock:  # 加锁保护
            # 如果是第一个点
            if not self.positions:
                self.positions.append(point.copy())
                self.current_point = point.copy()
                return

            # 如果已经有current_point，计算与current_point的距离
            if self.current_point is not None:
                distance = np.linalg.norm(point - self.current_point)
                self.accumulated_distance += distance
                self.current_point = point.copy()

                # 如果累积距离超过最小距离，将current_point添加为有效点
                if self.accumulated_distance >= self.min_distance:
                    self.positions.append(self.current_point.copy())
                    self.accumulated_distance = 0.0  # 重置累积距离

                    # 如果超过最大段数，移除最早的点
                    if len(self.positions) > self.max_segments + 1:
                        self.positions.pop(0)

                    # 尝试进行轨迹简化
                    if self.auto_simplify:
                        self._auto_simplify_trajectory()

    def clean_dense_points(self, target_distance=None):
        """
        清理轨迹中过于密集的点。

        参数：
        - target_distance: 目标距离，如果不指定则使用 min_distance
        """
        if target_distance is None:
            target_distance = self.min_distance

        if len(self.positions) < 2:
            return

        i = 0
        while i < len(self.positions) - 1:
            j = i + 1
            while j < len(self.positions):
                distance = np.linalg.norm(
                    np.array(self.positions[j]) - np.array(self.positions[i])
                )
                if distance < target_distance:
                    self.positions.pop(j)
                else:
                    i = j
                    break
            i += 1

    def draw_trajectory(
        self,
        viewer,
        color=[0, 1, 0, 1],
        width=0.002,
        fade=False,
        dynamic_color=None,
        dynamic_width=None,
    ):
        """
        绘制完整轨迹。

        参数：
        - viewer: Mujoco 查看器对象
        - color: 基础颜色 [r,g,b,a]
        - width: 线条宽度
        - fade: 是否启用渐变效果
        - dynamic_color: 动态轨迹段的颜色，如果不指定则与 color 相同
        - dynamic_width: 动态轨迹段的宽度，如果不指定则与 width 相同
        """
        with self.positions_lock:  # 加锁保护
            if len(self.positions) < 2:  # 修改判断条件
                return

            # 如果没有指定动态轨迹的参数，使用与基础轨迹相同的参数
            if dynamic_color is None:
                dynamic_color = color
            if dynamic_width is None:
                dynamic_width = width

            # 绘制固定轨迹
            for i in range(len(self.positions) - 1):
                if fade:
                    alpha = (i / (len(self.positions) - 1)) * color[3]
                else:
                    alpha = color[3]

                current_color = [color[0], color[1], color[2], alpha]

                try:
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        width=width,
                        from_=self.positions[i],
                        to=self.positions[i + 1],
                    )
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = current_color
                    viewer.user_scn.ngeom += 1
                except IndexError:
                    break  # 如果发生索引错误就退出循环

            # 绘制动态轨迹段（从最后一个固定点到当前点）
            if self.current_point is not None and len(self.positions) > 0:
                try:
                    mujoco.mjv_connector(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_LINE,
                        width=dynamic_width,
                        from_=self.positions[-1],
                        to=self.current_point,
                    )
                    viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = dynamic_color
                    viewer.user_scn.ngeom += 1
                except IndexError:
                    pass  # 忽略动态段的绘制错误

    def _auto_simplify_trajectory(self):
        """
        自动简化轨迹的内部方法。
        使用滑动窗口方式处理历史轨迹，保持最新段动态特性。
        """
        # 如果轨迹点数量不足，直接返回
        if len(self.positions) < self.min_points_to_simplify:
            return

        # 保留最新的一段轨迹不参与简化
        dynamic_segment = 1
        static_length = len(self.positions) - dynamic_segment

        # 确保有足够的新点进行处理
        if (static_length - self.last_simplified_index) < self.window_size // 2:
            return

        # 计算当前处理窗口的范围,增加重叠区域
        overlap = self.window_size // 4  # 添加25%的重叠
        window_start = max(0, static_length - self.window_size)
        window_end = static_length

        # 如果不是第一个窗口,则包含上一个窗口的一部分
        if window_start > 0:
            window_start = max(0, window_start - overlap)

        # 提取窗口内的点进行简化
        window_points = np.array(self.positions[window_start:window_end])

        # 使用更保守的简化参数
        local_epsilon = self.rdp_epsilon * 0.8  # 降低局部简化阈值
        simplified_points = RDPSimplifier.simplify(
            window_points, local_epsilon
        ).tolist()

        # 如果是重叠区域,保留前一个窗口的结果
        if window_start > 0:
            simplified_points = simplified_points[overlap:]

        # 更新轨迹,保持动态段不变
        self.positions = (
            self.positions[: window_start + (overlap if window_start > 0 else 0)]
            + simplified_points
            + self.positions[window_end:]
        )

        # 更新上次简化位置
        self.last_simplified_index = window_start + len(simplified_points)

    def simplify_trajectory(self, epsilon=None):
        """
        手动触发使用Ramer-Douglas-Peucker算法简化整条轨迹。

        参数：
        - epsilon: 简化阈值,值越大简化程度越高。如果不指定则使用类的默认值
        """
        if len(self.positions) < 3:
            return

        if epsilon is None:
            epsilon = self.rdp_epsilon

        positions = np.array(self.positions)
        self.positions = RDPSimplifier.simplify(positions, epsilon).tolist()


class TimeSeriesPlotter:
    """传感器数据图表管理器，支持多图表和多数据线"""

    def __init__(self, viewer, model, logger=None, y_range=None, x_range=(-100, 0)):
        """
        初始化图表管理器

        Args:
            viewer: MujocoViewer实例
            model: MuJoCo模型
            logger: 日志记录器
            y_range: 默认的Y轴范围，格式为(min, max)
            x_range: 默认的X轴范围，格式为(min, max)，默认为(-100, 0)
        """
        self.viewer = viewer
        self.model = model
        self.default_y_range = y_range  # 保存默认Y轴范围
        self.default_x_range = x_range  # 保存默认X轴范围

        if logger is None:
            logger_manager = setup_logger(
                log_file_prefix="sensorPlotter", context_names=["sensor"]
            )
            self._logger = logger_manager.get_logger("sensor")
        else:
            self._logger = logger

        self.figures = {}

    def _init_figure(self, fig_idx, title=None, y_range=None, x_range=None):
        """初始化图表配置"""
        try:
            fig = self.viewer.figs[fig_idx]
            fig.title = title if title else "Sensor Data"
            fig.xlabel = "Time (s)"

            # 设置X轴范围
            x_range = x_range or self.default_x_range
            fig.range[0][0] = x_range[0]  # X轴最小值
            fig.range[0][1] = x_range[1]  # X轴最大值

            # 设置Y轴范围
            if y_range:
                fig.range[1][0] = y_range[0]  # Y轴最小值
                fig.range[1][1] = y_range[1]  # Y轴最大值

            fig.gridsize[0] = 5
            fig.gridsize[1] = 5
            fig.gridwidth = 1
            fig.gridrgb = [0.2, 0.2, 0.2]
            fig.flg_extend = 0  # 禁用自动扩展，使用固定范围
            fig.flg_legend = 1

            self.figures[fig_idx] = {
                "title": title,
                "lines": {},
                "y_range": y_range,
                "x_range": x_range,
            }

            self._logger.info(
                f"已初始化图表 (fig_idx={fig_idx}, x_range={x_range}, y_range={y_range})"
            )

        except Exception as e:
            self._logger.error(f"初始化图表失败: {str(e)}")
            raise

    def create_figure(self, fig_idx=None, title=None, y_range=None, x_range=None):
        """创建新的图表"""
        fig_idx = fig_idx if fig_idx is not None else len(self.figures)
        y_range = y_range or self.default_y_range  # 使用传入的范围或默认范围
        x_range = x_range or self.default_x_range  # 使用传入的范围或默认范围
        self._init_figure(fig_idx, title, y_range, x_range)
        return fig_idx

    def add_sensor(self, sensor_name, fig_idx=None, line_name=None, y_range=None):
        """添加传感器数据线"""
        try:
            if fig_idx is None or fig_idx not in self.figures:
                y_range = y_range or self.default_y_range
                fig_idx = self.create_figure(fig_idx, y_range=y_range)

            # 获取传感器量程
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if sensor_id == -1:
                raise ValueError(f"找到传感器: {sensor_name}")

            # 使用传感器量程或指定范围
            if y_range is None and self.default_y_range is None:
                sensor_cutoff = self.model.sensor_cutoff[sensor_id]
                y_range = (0, sensor_cutoff)  # 使用 0 到 cutoff 的范围

            # 设置Y轴范围
            if y_range:
                fig = self.viewer.figs[fig_idx]
                fig.range[1][0] = y_range[0]
                fig.range[1][1] = y_range[1]
                fig.flg_extend = 0  # 禁用自动扩展

            # 添加数据线
            line_name = line_name or sensor_name
            self.viewer.add_line_to_fig(line_name, fig_idx=fig_idx)
            self.figures[fig_idx]["lines"][line_name] = sensor_name

            # 更新图例
            line_idx = len(self.figures[fig_idx]["lines"]) - 1
            fig.linename[line_idx] = line_name.encode("utf-8")

            self._logger.info(
                f"已添加传感器 {sensor_name} 的数据线 '{line_name}' 到图表 {fig_idx}"
            )

        except Exception as e:
            self._logger.error_rate_limited(f"添加传感器数据线失败: {str(e)}", 0.5)

    def update(self, sensor_data):
        """更新所有图表数据"""
        try:
            for fig_idx, fig_info in self.figures.items():
                for line_name, sensor_name in fig_info["lines"].items():
                    if sensor_name in sensor_data:
                        self.viewer.add_data_to_line(
                            line_name, sensor_data[sensor_name], fig_idx=fig_idx
                        )
        except Exception as e:
            self._logger.error_rate_limited(f"更新图表数据失败: {str(e)}", 0.5)

    def add_sensor_with_figure(
        self, sensor_name, title=None, line_name=None, y_range=None
    ):
        """创建新图表并添加传感器的便捷方法"""
        title = title or sensor_name
        y_range = y_range or self.default_y_range  # 使用传入的范围或默认范围
        fig_idx = self.create_figure(title=title, y_range=y_range)
        self.add_sensor(
            sensor_name, fig_idx=fig_idx, line_name=line_name, y_range=y_range
        )
        return fig_idx


class SeamProfilePlotter:
    """焊缝轮廓实时绘制器，用于显示激光阵列传感器数据"""

    DEFAULT_CONFIG = {
        # 基础配置
        "sensor_prefix": "line_laser",  # 传感器名称前缀
        "y_range": None,  # Y轴范围，None时自动从传感器获取
        "data_points": 1000,  # 预设数据点数量
        # 视觉配置
        "line_color": [0, 1, 0],  # 线条颜色 RGB
        "line_alpha": 1.0,  # 线条透明度
        "line_width": 1,  # 线条宽度
        "grid_size": [5, 5],  # 网格大小 [x, y]
        "grid_width": 1,  # 网格线宽度
        "grid_color": [0.2, 0.2, 0.2],  # 网格颜色
        # 图表标题
        "title": "Seam Profile",
        "xlabel": "Sensor Index",
        # 新增焊枪标记配置
        "torch_color": [1, 0, 0],  # 焊枪标记颜色(红色)
        "torch_width": 2,  # 焊枪标记线宽
        "torch_height": 0.04,  # 焊枪标记线长度
        # 新增焊缝居中配置
        "center_seam": True,  # 是否将焊缝居中显示
        "seam_window": 0.02,  # 焊缝显示窗口大小
        # 添加新的配置项
        "min_value_threshold": 0.0001,  # 判断最小值的阈值
        "torch_sensor_index": None,  # 不再需要这个配置，焊枪总是对应中间传感器
    }

    def __init__(self, viewer, model, logger=None, config=None, sensor_count=None):
        """
        初始化焊缝轮廓绘制器

        Args:
            viewer: MujocoViewer实例
            model: MuJoCo模型
            logger: 日志记录器
            config: 配置字典，用于覆盖默认配置
            sensor_count: 传感器数量,如果不指定则自动检测
        """
        self.viewer = viewer
        self.model = model

        # 合并配置
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # 设置日志器
        if logger is None:
            logger_manager = setup_logger(
                log_file_prefix="seamPlotter", context_names=["seam"]
            )
            self._logger = logger_manager.get_logger("seam")
        else:
            self._logger = logger

        # 获取传感器数量
        self.sensor_count = (
            sensor_count if sensor_count is not None else self._get_sensor_count()
        )
        self._logger.info(f"SeamProfilePlotter 检测到 {self.sensor_count} 个激光传感器")

        # 生成传感器名称列表
        self.sensor_names = SensorNameGenerator.generate_names(
            prefix=self.config["sensor_prefix"], count=self.sensor_count
        )

        # 获取Y轴范围
        if self.config["y_range"] is None:
            self.config["y_range"] = self._get_sensor_range()

        # 创建图表
        self.fig_idx = len(self.viewer.figs)
        self._init_figure()

        # 新增属性
        self.torch_position = 0.0  # 焊枪在x轴上的位置
        self.seam_center = 0.0  # 焊缝中心位置
        self.last_valid_data = {}  # 存储上一帧有效的传感器数据

        # 焊枪对应中间传感器
        self.center_sensor_index = self.sensor_count // 2

    def _get_sensor_count(self):
        """自动计算激光传感器的数量"""
        count = 0
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name and name.startswith(self.config["sensor_prefix"]):
                count += 1

        if count == 0:
            raise ValueError(f"未找到前缀为 {self.config['sensor_prefix']} 的传感器")

        return count

    def _get_sensor_range(self):
        """从传感器获取合适的Y轴范围"""
        try:
            sensor_name = f"{self.config['sensor_prefix']}00"
            sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
            )
            if sensor_id != -1:
                cutoff = self.model.sensor_cutoff[sensor_id]
                return (-cutoff, 0)
        except Exception as e:
            self._logger.warning(f"获取传感器范围失败: {str(e)}")

        return (-0.08, 0)  # 默认范围

    def _init_figure(self):
        """初始化图表配置"""
        try:
            # 创建新图表
            fig = mujoco.MjvFigure()
            mujoco.mjv_defaultFigure(fig)

            # 设置基本属性
            fig.title = self.config["title"].encode("utf-8")
            fig.xlabel = self.config["xlabel"].encode("utf-8")

            # 设置范围
            fig.range[0][0] = 0  # X轴最小值
            fig.range[0][1] = self.sensor_count  # X轴最大值
            fig.range[1][0] = self.config["y_range"][0]  # Y轴最小值
            fig.range[1][1] = self.config["y_range"][1]  # Y轴最大值

            # 网格设置
            fig.gridsize[0] = self.config["grid_size"][0]
            fig.gridsize[1] = self.config["grid_size"][1]
            fig.gridwidth = self.config["grid_width"]
            fig.gridrgb = self.config["grid_color"]

            # 其他设置
            fig.flg_extend = 0  # 禁用自动扩展
            fig.flg_legend = 0  # 禁用图例

            # 设置线条
            self.line_name = "seam_profile"
            fig.linename[0] = self.line_name.encode("utf-8")

            # 设置线条颜色
            color = self.config["line_color"]
            fig.linergb[0] = [color[0], color[1], color[2]]

            # 初始化数据点
            for i in range(self.config["data_points"]):
                fig.linedata[0][2 * i] = float(i)
                fig.linedata[0][2 * i + 1] = 0.0

            self.viewer.figs.append(fig)
            self._logger.info("焊缝轮廓绘制器初始化成功")

        except Exception as e:
            self._logger.error(f"初始化图表失败: {str(e)}")
            raise

    def set_torch_position(self, position):
        """
        设置焊枪位置

        Args:
            position: 仿真世界中的焊枪 x 坐标（这个值我们不直接使用）
        """
        # 焊枪始终对应中间传感器的位置
        self.torch_position = self.center_sensor_index

    def _draw_torch_marker(self, fig, x_position):
        """
        绘制焊枪位置标记

        Args:
            fig: 图表对象
            x_position: 焊枪在图表中的x坐标
        """
        if not hasattr(self, "torch_line_idx"):
            # 添加焊枪标记线
            self.torch_line_idx = 1  # 使用第二条线
            fig.linename[self.torch_line_idx] = b"torch"
            fig.linergb[self.torch_line_idx] = self.config["torch_color"]

        # 计算焊枪标记的起点和终点
        y_range = self.config["y_range"]
        height = self.config["torch_height"]
        mid_y = (y_range[0] + y_range[1]) / 2

        # 绘制竖直线
        fig.linepnt[self.torch_line_idx] = 2
        fig.linedata[self.torch_line_idx][0] = x_position
        fig.linedata[self.torch_line_idx][1] = mid_y - height / 2
        fig.linedata[self.torch_line_idx][2] = x_position
        fig.linedata[self.torch_line_idx][3] = mid_y + height / 2

    def _find_seam_center(self, sensor_data):
        """计算焊缝中心位置，考虑多个接近的最小值"""
        if not sensor_data:
            return self.seam_center  # 如果没有新数据，保持上一次的中心位置

        # 第一遍扫描找到全局最小值
        min_value = float("inf")
        for sensor_name in self.sensor_names:
            if sensor_name in sensor_data:
                value = sensor_data[sensor_name]
                if value is not None and value < min_value:
                    min_value = value

        # 第二遍扫描找到所有接近最小值的点
        min_indices = []
        threshold = self.config["min_value_threshold"]
        for i, sensor_name in enumerate(self.sensor_names):
            if sensor_name in sensor_data:
                value = sensor_data[sensor_name]
                if value is not None and abs(value - min_value) <= threshold:
                    min_indices.append(i)

        # 如果有多个最小值点，取中值位置
        if min_indices:
            return float(min_indices[len(min_indices) // 2])
        else:
            return self.seam_center  # 如果没找到最小值，保持上一次的中心位置

    def update(self, sensor_data):
        """更新焊缝轮廓图表"""
        try:
            fig = self.viewer.figs[self.fig_idx]

            # 更新有效数据
            if sensor_data:
                self.last_valid_data = sensor_data.copy()
            else:
                sensor_data = self.last_valid_data

            # 计算焊逢中心
            if self.config["center_seam"]:
                self.seam_center = self._find_seam_center(sensor_data)

            # 重置数据点计数
            fig.linepnt[0] = 0

            # 收集有效的传感器数据
            valid_points = []  # 存储有效的数据点
            for i, sensor_name in enumerate(self.sensor_names):
                if sensor_name in sensor_data:
                    value = sensor_data[sensor_name]
                    if value is not None:
                        # 计算相对于中心的x坐标
                        if self.config["center_seam"]:
                            x = float(i) - self.seam_center + self.sensor_count / 2
                        else:
                            x = float(i)
                        valid_points.append((x, value, i))

            # 按x坐标排序
            valid_points.sort(key=lambda p: p[0])

            # 确定实际的显示范围
            display_min = 0
            display_max = self.sensor_count

            all_points = []

            # 补充左边界数据(仅在确实缺失时)
            if (
                valid_points and valid_points[0][0] > display_min + 0.5
            ):  # 添加一个小的容差
                left_value = valid_points[0][1]
                left_start = max(display_min, int(valid_points[0][0] - 1))
                for x in range(left_start, -1, -1):
                    all_points.append((float(x), left_value))

            # 添加有效数据点
            for x, value, _ in valid_points:
                all_points.append((x, value))

            # 补充右边界数据(仅在确实缺失时)
            if (
                valid_points and valid_points[-1][0] < display_max - 0.5
            ):  # 添加一个小的容差
                right_value = valid_points[-1][1]
                right_start = min(display_max, int(valid_points[-1][0] + 1))
                for x in range(right_start, self.sensor_count + 1):
                    all_points.append((float(x), right_value))

            # 按x坐标排序并写入所有数据点
            all_points.sort(key=lambda p: p[0])
            for x, value in all_points:
                if display_min <= x <= display_max:  # 确保在显示范围内
                    idx = fig.linepnt[0] * 2
                    fig.linedata[0][idx] = x
                    fig.linedata[0][idx + 1] = value
                    fig.linepnt[0] += 1

            # 绘制焊枪位置标记
            if self.config["center_seam"]:
                torch_x = self.torch_position - self.seam_center + self.sensor_count / 2
            else:
                torch_x = self.torch_position

            self._draw_torch_marker(fig, torch_x)

        except Exception as e:
            self._logger.error_rate_limited(f"更新焊缝轮廓数据失败: {str(e)}", 0.5)
