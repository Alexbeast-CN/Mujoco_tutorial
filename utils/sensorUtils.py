"""
MuJoCo传感器工具库
具体信息请查看 [SensorUtils 设计说明](docs/sensorUtils设计说明.md)
"""

import mujoco
from typing import Optional, List, NamedTuple
import numpy as np
import threading
from pathlib import Path
import time
from datetime import datetime
import csv
from abc import ABC, abstractmethod
import math
from utils.logUtils import RateLimitedLogger, setup_logger
import os


class SensorNameGenerator:
    """传感器名称生成器"""

    @staticmethod
    def _get_digit_width(count: int) -> int:
        """
        根据数量确定需要的位数

        Args:
            count: 传感器数量

        Returns:
            int: 需要的位数(1-4)
        """
        if count <= 0:
            return 1
        return len(str(count))

    @staticmethod
    def validate_count(count: int) -> None:
        """
        验证传感器数量是否合法

        Args:
            count: 传感器数量

        Raises:
            ValueError: 当数量为负数时抛出
        """
        if count < 0:
            raise ValueError(f"传感器数量不能为负数: {count}")

    @staticmethod
    def validate_prefix(prefix: str) -> None:
        """
        验证前缀是否合法

        Args:
            prefix: 传感器名称前缀

        Raises:
            ValueError: 当前缀为空或包含非法字符时抛出
        """
        if not prefix:
            raise ValueError("前缀不能为空")
        if not prefix.replace("_", "").isalnum():
            raise ValueError(f"前缀包含非法字符: {prefix}")

    @classmethod
    def generate_names(
        cls,
        prefix: str,
        count: int,
        separator: str = "",
        custom_format: Optional[str] = None,
    ) -> List[str]:
        """
        生成传感器名称列表

        Args:
            prefix: 传感器名称前缀
            count: 传感器数量
            separator: 前缀和编号之间的分隔符(默认为空)
            custom_format: 自定义的格式化字符串(可选)

        Returns:
            List[str]: 传感器名称列表

        Examples:
            >>> SensorNameGenerator.generate_names("sensor", 5)
            ['sensor0', 'sensor1', 'sensor2', 'sensor3', 'sensor4']
            >>> SensorNameGenerator.generate_names("laser", 10)
            ['laser00', 'laser01', ..., 'laser09']
            >>> SensorNameGenerator.generate_names("dev", 150)
            ['dev000', 'dev001', ..., 'dev149']
        """
        # 参数验证
        cls.validate_prefix(prefix)
        cls.validate_count(count)

        # 确定位数
        digit_width = cls._get_digit_width(count)

        # 使用自定义格式或根据位数生成格式
        if custom_format:
            format_str = custom_format
        else:
            format_str = f"{{:0{digit_width}d}}"

        # 生成名称列表
        return [f"{prefix}{separator}{format_str.format(i)}" for i in range(count)]

    @classmethod
    def generate_range_names(
        cls, prefix: str, start: int, end: int, **kwargs
    ) -> List[str]:
        """
        生成指定范围的传感器名称

        Args:
            prefix: 传感器名称前缀
            start: 起始编号(包含)
            end: 结束编号(不包含)
            **kwargs: 其他参数传递给generate_names

        Returns:
            List[str]: 传感器名称列表

        Raises:
            ValueError: 当start大于等于end时抛出
        """
        # 验证前缀
        cls.validate_prefix(prefix)

        if start >= end:
            raise ValueError(f"起始编号必须小于结束编号: {start} >= {end}")

        count = end - start
        digit_width = cls._get_digit_width(end)  # 使用end决定位数
        format_str = f"{{:0{digit_width}d}}"

        return [
            f"{prefix}{kwargs.get('separator', '')}{format_str.format(i)}"
            for i in range(start, end)
        ]


class ThreadedSensor:
    """传感器线程装饰器"""

    def __init__(self, sensor):
        self.sensor = sensor
        self.sensor.threaded = True  # 设置传感器的 threaded 标志
        self._thread = None
        self._running = False
        self._last_read_time = 0
        # 添加帧率统计相关属性
        self._frame_count = 0
        self._start_time = None

    def start(self):
        """启动传感器线程"""
        if self._thread is not None:
            self.sensor._logger.warning("传感器线程已在运行")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        # 重置计数
        self._frame_count = 0
        self._start_time = time.time()
        self._thread.start()
        self.sensor._logger.info(
            f"{self.sensor.__class__.__name__} 传感器线程已启动,采样频率: {self.sensor.frequency}Hz"
        )

    def stop(self):
        """停止传感器线程"""
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            self._start_time = None  # 清除开始时间
            self.sensor._logger.info("传感器线程已停止")

    def _run(self):
        period = 1.0 / self.sensor.frequency
        while self._running:
            current_time = time.time()
            elapsed = current_time - self._last_read_time

            if elapsed >= period:
                try:
                    self.sensor._read()
                    self._last_read_time = current_time
                    self._frame_count += 1  # 增加帧计数
                except Exception as e:
                    self.sensor._logger.error(f"传感器读取失败: {str(e)}")

            sleep_time = max(0, period - (time.time() - current_time))
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_frame_rate(self) -> float:
        """计算当前帧率"""
        if not self._running or self._start_time is None:
            return 0.0

        elapsed_time = time.time() - self._start_time
        if elapsed_time > 0:
            return self._frame_count / elapsed_time
        return 0.0

    def recording_on(self, data_dir: str = "recordings"):
        """开始记录传感器数据"""
        return self.sensor.recording_on(data_dir)

    def recording_off(self):
        """停止记录传感器数据"""
        return self.sensor.recording_off()

    # 代理原始传感器的属性和方法
    def __getattr__(self, name):
        """代理所有未实现的属性和方法到原始传感器"""
        return getattr(self.sensor, name)


class BaseSensor(ABC):
    """单传感器基类"""

    def __init__(self, model, data, sensor_name: str, frequency=100, logger=None):
        """
        初始化传感器基类

        Args:
            model: MuJoCo模型对象
            data: MuJoCo数据对象
            sensor_name: 传感器名称
            frequency: 传感器采样频率(Hz)
            logger: 日志记录器,为None时将自动创建
        """
        self.model = model
        self.data = data
        self.sensor_name = sensor_name
        self.frequency = frequency
        self._setup_logger(logger)

        # 获取传感器ID
        self.sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, self.sensor_name
        )
        if self.sensor_id == -1:
            raise ValueError(f"未找到传感器: {self.sensor_name}")

        self._latest_data = None
        self._recording = False
        self._record_file = None
        self._record_writer = None
        self._record_start_time = None
        self.threaded = False  # 添加 threaded 标志

    def _setup_logger(self, logger):
        """初始化日志记录器"""
        if logger is None:
            logger_manager = setup_logger(
                log_file_prefix=self.__class__.__name__.lower(),
                context_names=[self.__class__.__name__.lower()],
            )
            self._logger = logger_manager.get_logger(self.__class__.__name__.lower())
            self._logger.warning(
                f"未提供logger,已自动创建新的{self.__class__.__name__.lower()} logger"
            )
        else:
            self._logger = logger

    def recording_on(self, data_dir: str = "recordings"):
        """开始记录传感器数据"""
        if self._recording:
            self._logger.warning("传感器已经在记录中")
            return False

        try:
            data_path = Path(data_dir)
            # 检查目录是否存在且有写权限
            if not data_path.exists():
                try:
                    data_path.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError) as e:
                    self._logger.error(f"创建目录失败: {str(e)}")
                    return False
            elif not os.access(str(data_path), os.W_OK):
                self._logger.error(f"目录无写入权限: {data_dir}")
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = data_path / f"{self.sensor_name}_{timestamp}.csv"

            self._record_file = open(file_path, "w", newline="")
            self._record_writer = csv.writer(self._record_file)
            self._record_writer.writerow(["timestamp", "value"])
            self._record_start_time = time.time()
            self._recording = True

            self._logger.info(f"开始记录传感器数据到文件: {file_path}")
            return True

        except Exception as e:
            self._logger.error(f"开始录制失败: {str(e)}")
            return False

    def recording_off(self):
        """停止记录传感器数据"""
        if not self._recording:
            return True

        try:
            self._record_file.close()
            self._logger.info("停止记录传感器数据")
            return True
        except Exception as e:
            self._logger.error(f"停止记录失败: {str(e)}")
            return False
        finally:
            self._recording = False
            self._record_file = None
            self._record_writer = None
            self._record_start_time = None

    def _record_value(self, value: float):
        """记录传感器数据"""
        if not self._recording or value is None:
            return

        try:
            current_time = time.time()
            relative_time = current_time - self._record_start_time
            self._record_writer.writerow([f"{relative_time:.6f}", f"{value:.6f}"])
        except Exception as e:
            self._logger.error(f"写入数据失败: {str(e)}")
            self.recording_off()

    @property
    def reading(self):
        """获取最新的传感器读数"""
        # 根据 threaded 状态决定是否调用 _read()
        if not self.threaded:
            self._latest_data = self._read()
        return self._latest_data

    @abstractmethod
    def _read(self):
        """读取传感器数据的抽象方法"""
        pass


class LaserScanPoint(NamedTuple):
    """激光扫描点"""

    distance: float  # 测量距离
    position: np.ndarray  # 世界坐标系中的位置 (x, y, z)
    site: np.ndarray  # 传感器位置 (x, y, z)
    direction: np.ndarray  # 测量方向 (归一化向量)
    valid: bool  # 测量是否有效


class RangeFinder(BaseSensor):
    """单个距离传感器"""

    def __init__(
        self,
        model,
        data,
        sensor_name: str,
        site_name: str = None,
        frequency=100,
        logger=None,
    ):
        """
        初始化距离传感器

        Args:
            model: MuJoCo模型对象
            data: MuJoCo数据对象
            sensor_name: 传感器名称
            site_name: 传感器安装位置的site名称(可选)
            frequency: 传感器采样频率(Hz)
            logger: 日志记录器
        """
        super().__init__(model, data, sensor_name, frequency, logger)
        self._rng = np.random.default_rng()

        # 如果提供了site名称，获取site ID
        self.site_id = None
        if site_name:
            self.site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
            )
            if self.site_id == -1:
                raise ValueError(f"未找到site: {site_name}")

    def _read(self) -> float:
        """读取距离传感器数据"""
        try:
            value = self.data.sensordata[self.sensor_id]
            noise_std = self.model.sensor_noise[self.sensor_id]

            if noise_std > 0:
                value += self._rng.normal(0, noise_std)

            self._latest_data = value

            # 记录数据
            self._record_value(value)

            return value

        except Exception as e:
            self._logger.error(f"读取传感器数据失败: {str(e)}")
            return float("inf")

    def get_measurement_info(self) -> Optional[LaserScanPoint]:
        """
        获取传感器测量信息，包括测量距离和方向

        Returns:
            Optional[LaserScanPoint]: 传感器测量点信息，如果无法获取则返回None
        """
        if self.site_id is None:
            return None

        try:
            # 获取传感器位置和方向
            site_position = self.data.site_xpos[self.site_id].copy()
            rotation = self.data.site_xmat[self.site_id].reshape(3, 3)

            direction = rotation[:, 2]  # z轴方向

            # 获取测量距离
            distance = self.reading

            # 检查测量值是否有效
            valid = np.isfinite(distance) and distance > 0

            # 计算测量点在世界坐标系中的位置
            if valid:
                point_position = site_position + direction * distance
            else:
                # 对于无效点,直接使用传感器位置
                point_position = site_position

            return LaserScanPoint(
                distance=distance,
                position=point_position,
                site=site_position,
                direction=direction,
                valid=valid,
            )

        except Exception as e:
            self._logger.error(f"获取测量信息失败: {str(e)}")
            return None


class LaserScanFrame:
    """激光扫描帧，包含一次扫描的所有点数据"""

    def __init__(self, points: List[LaserScanPoint], timestamp: float = None):
        self.points = points
        self.timestamp = timestamp or time.time()

    @property
    def distances(self) -> np.ndarray:
        """返回所有测量距离"""
        return np.array([p.distance for p in self.points])

    @property
    def positions(self) -> np.ndarray:
        """返回所有点的位置"""
        return np.array([p.position for p in self.points])

    @property
    def directions(self) -> np.ndarray:
        """返回所有测量方向"""
        return np.array([p.direction for p in self.points])

    @property
    def valid_mask(self) -> np.ndarray:
        """返回有效点的掩码"""
        return np.array([p.valid for p in self.points])

    @property
    def valid_points(self) -> np.ndarray:
        """返回所有有效点的位置"""
        return self.positions[self.valid_mask]

    def get_points_in_range(self, min_dist: float, max_dist: float) -> np.ndarray:
        """返回指定距离范围内的点"""
        distances = self.distances
        mask = (distances >= min_dist) & (distances <= max_dist) & self.valid_mask
        return self.positions[mask]


class LaserScan:
    """激光扫描传感器阵列"""

    def __init__(
        self,
        model,
        data,
        frequency=50,
        logger=None,
        sensor_prefix: str = "line_laser",
        site_prefix: str = "laser_array",
    ):
        self.model = model
        self.data = data
        self.frequency = frequency
        self._logger = logger
        self.sensor_prefix = sensor_prefix
        self.site_prefix = site_prefix

        # 初始化传感器
        self.sensor_count = 0
        self.sensor_names = []
        self.site_names = []
        self._init_sensors()

        # 内部状态
        self._latest_frame = None
        self._recording = False
        self._record_file = None
        self._record_writer = None
        self._record_start_time = None
        self.threaded = False

    def _init_sensors(self):
        """初始化传感器配置"""
        # 获取传感器数量
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name and name.startswith(self.sensor_prefix):
                self.sensor_count += 1

        if self.sensor_count == 0:
            raise ValueError(f"未找到前缀为 {self.sensor_prefix} 的传感器")

        self._logger.info(f"LaserScan 找到 {self.sensor_count} 个激光传感器")

        # 生成传感器和site名称
        self.sensor_names = SensorNameGenerator.generate_names(
            prefix=self.sensor_prefix, count=self.sensor_count
        )
        self.site_names = SensorNameGenerator.generate_names(
            prefix=self.site_prefix, count=self.sensor_count
        )

        # 为每个传感器创建RangeFinder实例
        self.sensors = [
            RangeFinder(
                model=self.model,
                data=self.data,
                sensor_name=sensor_name,
                site_name=site_name,
                frequency=self.frequency,
                logger=self._logger,
            )
            for sensor_name, site_name in zip(self.sensor_names, self.site_names)
        ]

    def _read(self) -> LaserScanFrame:
        """同步读取所有传感器数据"""
        points = []
        timestamp = time.time()  # 记录采集时间戳

        # 同步读取所有传感器数据
        for sensor in self.sensors:
            point = sensor.get_measurement_info()
            if point is not None:
                points.append(point)

        frame = LaserScanFrame(points, timestamp)
        self._latest_frame = frame

        # 添加记录功能
        if self._recording:
            self._record_frame(frame)

        return frame

    @property
    def reading(self) -> Optional[LaserScanFrame]:
        """获取最新的传感器读数"""
        if not self.threaded:
            self._latest_frame = self._read()
        return self._latest_frame

    def recording_on(self, data_dir: str = "laser_scan_data"):
        """开始记录扫描数据"""
        if self._recording:
            self._logger.warning("激光扫描已在记录中")
            return False

        try:
            data_path = Path(data_dir)
            # 检查目录是否存在且有写权限
            if not data_path.exists():
                try:
                    data_path.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError) as e:
                    self._logger.error(f"创建目录失败: {str(e)}")
                    return False
            elif not os.access(str(data_path), os.W_OK):
                self._logger.error(f"目录无写入权限: {data_dir}")
                return False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = data_path / f"laser_scan_{timestamp}.csv"

            self._record_file = open(file_path, "w", newline="")
            self._record_writer = csv.writer(self._record_file)

            headers = ["timestamp"]
            for i in range(self.sensor_count):
                headers.extend([f"distance_{i}", f"x_{i}", f"y_{i}", f"z_{i}"])
            self._record_writer.writerow(headers)

            self._record_start_time = time.time()
            self._recording = True
            self._logger.info(f"开始记录扫描数据到文件: {file_path}")
            return True

        except Exception as e:
            self._logger.error(f"开始录制失败: {str(e)}")
            return False

    def recording_off(self):
        """停止记录扫描数据"""
        if not self._recording:
            return

        try:
            self._record_file.close()
            self._logger.info("停止记录扫描数据")
        except Exception as e:
            self._logger.error(f"停止记录失败: {str(e)}")
        finally:
            self._recording = False
            self._record_file = None
            self._record_writer = None
            self._record_start_time = None

    def _record_frame(self, frame: LaserScanFrame):
        """记录一帧数据"""
        if not self._recording or frame is None:
            return

        try:
            current_time = time.time()
            relative_time = current_time - self._record_start_time
            row = [relative_time]
            for point in frame.points:
                row.extend(
                    [
                        point.distance,
                        point.position[0],
                        point.position[1],
                        point.position[2],
                    ]
                )
            self._record_writer.writerow(row)

        except Exception as e:
            self._logger.error(f"写入扫描数据失败: {str(e)}")
            self.recording_off()


class LineLaserFrame(LaserScanFrame):
    """线激光扫描帧,包含二维坐标信息"""

    def __init__(
        self,
        points: List[LaserScanPoint],
        timestamp: float = None,
        plane_normal: np.ndarray = None,
        transform_matrix: np.ndarray = None,
        origin: np.ndarray = None,
    ):
        super().__init__(points, timestamp)
        self.plane_normal = plane_normal
        self._2d_positions = None
        # 保存坐标变换信息
        self.transform_matrix = transform_matrix
        self.origin = origin

    @property
    def positions_2d(self) -> np.ndarray:
        """返回所有点的二维坐标"""
        if self._2d_positions is None:
            # 延迟计算二维坐标
            return np.zeros((len(self.points), 2))
        return self._2d_positions

    @positions_2d.setter
    def positions_2d(self, value: np.ndarray):
        """设置二维坐标"""
        self._2d_positions = value

    def transform_to_3d(self, point_2d: np.ndarray) -> np.ndarray:
        """
        将二维平面坐标转换为三维世界坐标

        Args:
            point_2d: 二维平面坐标点 shape=(2,)

        Returns:
            np.ndarray: 三维世界坐标点 shape=(3,)

        Raises:
            ValueError: 如果缺少变换矩阵或原点信息,或输入shape不正确
        """
        if self.transform_matrix is None or self.origin is None:
            raise ValueError("缺少坐标变换信息")

        # 检查输入
        if point_2d.shape != (2,):
            raise ValueError(f"二维点的shape应该是(2,), 但得到了{point_2d.shape}")

        # 使用变换矩阵的前两行进行反变换
        point_3d = self.origin + np.dot(point_2d, self.transform_matrix[:2])

        return point_3d

    def transform_points_to_3d(self, points_2d: np.ndarray) -> np.ndarray:
        """
        批量将二维平面坐标转换为三维世界坐标

        Args:
            points_2d: 二维平面坐标点数组 shape=(N, 2)

        Returns:
            np.ndarray: 三维世界坐标点数组 shape=(N, 3)

        Raises:
            ValueError: 如果缺少变换矩阵或原点信息,或输入shape不正确
        """
        if self.transform_matrix is None or self.origin is None:
            raise ValueError("缺少坐标变换信息")

        # 检查输入
        if len(points_2d.shape) != 2 or points_2d.shape[1] != 2:
            raise ValueError(
                f"二维点数组的shape应该是(N, 2), 但得到了{points_2d.shape}"
            )

        # 批量转换
        points_3d = self.origin + np.dot(points_2d, self.transform_matrix[:2])

        return points_3d


class LineLaser(LaserScan):
    """线激光扫描器,所有传感器在同一平面上"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 验证传感器配置
        self.is_collinear = False
        if not self._validate_sensor_arrangement():
            raise ValueError("传感器排列不符合线激光要求")

        # 计算平面法向量和变换矩阵
        self.plane_normal = None
        self.transform_matrix = None
        self.origin = None
        self._setup_plane_and_transform()

        self._logger.info(f"线激光平面法向量: {self.plane_normal}")

    def _validate_sensor_arrangement(self) -> bool:
        """
        验证传感器排列是否合法
        先检查是否在同一直线上,如果不是再检查是否在同一平面上
        """
        if len(self.sensors) < 3:
            return False

        # 获取所有传感器的位置
        positions = []
        for sensor in self.sensors:
            if sensor.site_id is None:
                return False
            positions.append(self.data.site_xpos[sensor.site_id])
        positions = np.array(positions)

        # 检查是否共线
        p0 = positions[0]
        p1 = positions[1]
        line_dir = p1 - p0
        line_dir = line_dir / np.linalg.norm(line_dir)

        line_tolerance = 1e-6  # 共线判断的容差
        self.is_collinear = True

        for p in positions[2:]:
            # 点到直线的距离
            v = p - p0
            dist_to_line = np.linalg.norm(np.cross(line_dir, v))
            if dist_to_line > line_tolerance:
                self.is_collinear = False
                break

        if self.is_collinear:
            self._logger.info("传感器排列在同一直线上")
            return True

        # 如果不共线,检查是否共面
        v1 = positions[1] - p0
        v2 = positions[2] - p0

        # 计算平面法向量
        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)

        if normal_length < 1e-10:
            self._logger.warning("无法确定平面法向量")
            return False

        normal = normal / normal_length

        # 检查所有点到平面的距离
        plane_tolerance = 1e-6  # 共面判断的容差
        is_coplanar = True

        for p in positions[3:]:
            # 点到平面的距离
            distance = abs(np.dot(p - p0, normal))
            if distance > plane_tolerance:
                self._logger.warning(f"检测到非共面的传感器,距离平面: {distance}")
                is_coplanar = False
                break

        if is_coplanar:
            self._logger.info("传感器排列在同一平面上")
            return True

        return False

    def _setup_plane_and_transform(self) -> None:
        """计算激光平面的法向量和坐标变换矩阵"""
        # 获取第一个传感器的位置和测量方向
        p0 = self.data.site_xpos[self.sensors[0].site_id]
        sensor_dir = self.sensors[0].get_measurement_info().direction

        # 设置原点
        self.origin = p0

        # x轴方向为第一个传感器的测量方向
        y_axis = sensor_dir

        if self.is_collinear:
            # 如果传感器共线,使用传感器排列方向和测量方向
            p1 = self.data.site_xpos[self.sensors[1].site_id]
            line_dir = p1 - p0
            normal = np.cross(line_dir, sensor_dir)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                self._logger.warning("无法确定平面法向量,使用默认值")
                self.plane_normal = np.array([0, 0, 1])
            else:
                self.plane_normal = normal / norm

        else:
            # 如果不共线,使用三点确定平面
            p1 = self.data.site_xpos[self.sensors[1].site_id]
            p2 = self.data.site_xpos[self.sensors[2].site_id]

            v1 = p1 - p0
            v2 = p2 - p0
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)

            if norm < 1e-10:
                self._logger.warning("无法确定平面法向量,使用默认值")
                self.plane_normal = np.array([0, 0, 1])
            else:
                self.plane_normal = normal / norm

        # 计算y轴方向为法向量和x轴的叉积
        x_axis = np.cross(self.plane_normal, y_axis)
        x_axis = -x_axis / np.linalg.norm(x_axis)

        # 构建变换矩阵
        self.transform_matrix = np.array([x_axis, y_axis, self.plane_normal])

    def _transform_to_2d(self, point_3d: np.ndarray) -> np.ndarray:
        """将三维点转换为二维平面坐标"""
        # 平移到原点
        centered = point_3d - self.origin

        # 投影到xy平面
        point_2d = np.dot(centered, self.transform_matrix[:2].T)

        return point_2d

    def _read(self) -> LineLaserFrame:
        """读取一帧数据并计算二维坐标"""
        # 获取三维数据
        frame_3d = super()._read()

        # 转换为LineLaserFrame，传入变换信息
        frame = LineLaserFrame(
            points=frame_3d.points,
            timestamp=frame_3d.timestamp,
            plane_normal=self.plane_normal,
            transform_matrix=self.transform_matrix,
            origin=self.origin,
        )

        # 计算二维坐标
        positions_2d = np.array(
            [self._transform_to_2d(point.position) for point in frame.points]
        )
        frame.positions_2d = positions_2d

        self._latest_frame = frame
        return frame
