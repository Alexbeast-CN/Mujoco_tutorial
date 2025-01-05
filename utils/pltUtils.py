import matplotlib

matplotlib.use("TkAgg")  # 在导入 pyplot 之前设置后端

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import time
from typing import Callable, Optional, Tuple, Any
from collections import deque


class LidarPlotter:
    """用于可视化激光雷达类型传感器数据的类"""

    def __init__(
        self,
        num_sensors: int = 36,
        angle_range: Tuple[float, float] = (0, 360),
        max_range: float = 1.0,
        update_interval: int = 30,
    ):
        """
        初始化雷达可视化器

        参数:
            num_sensors: 传感器数量
            angle_range: 角度范围(起始角度,结束角度)
            max_range: 最大测量范围
            update_interval: 更新间隔(毫秒)
        """
        self.num_sensors = num_sensors
        self.angle_range = angle_range
        self.max_range = max_range
        self.update_interval = update_interval

        # 初始化图形
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.ax.set_title("Sensor Visualization")
        self.ax.set_theta_direction(-1)
        self.ax.set_theta_zero_location("N")
        (self.line,) = self.ax.plot([], [], "r.")

        # FPS计数器
        self.fps_text = self.ax.text(0, 1.1, "", transform=self.ax.transAxes)
        self.last_time = time.time()
        self.frame_count = 0

        # 计算角度数组
        angle_step = (angle_range[1] - angle_range[0]) / num_sensors
        self.angles = np.radians(np.arange(angle_range[0], angle_range[1], angle_step))

        # 动画对象
        self.anim = None

    def _init(self):
        """初始化动画"""
        self.line.set_data([], [])
        return self.line, self.fps_text

    def _update(self, frame: int, update_func: Callable):
        """更新动画帧"""
        # 更新FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.last_time)
            self.fps_text.set_text(f"FPS: {fps:.1f}")
            self.last_time = current_time

        # 获取新的传感器数据
        ranges = update_func()

        # 更新图形
        self.line.set_data(self.angles, ranges)
        self.ax.set_rmax(self.max_range)

        return self.line, self.fps_text

    def start(self, update_func: Callable, close_callback: Optional[Callable] = None):
        """
        启动可视化

        参数:
            update_func: 更新函数,接收frame参数,返回传感器数据数组
            close_callback: 窗口关闭时的回调函数
        """
        # 设置关闭事件处理
        if close_callback:
            self.fig.canvas.mpl_connect(
                "close_event", lambda event: (close_callback(), self.close())
            )

        # 创建动画
        self.anim = FuncAnimation(
            self.fig,
            lambda frame: self._update(frame, update_func),
            init_func=self._init,
            frames=None,
            interval=self.update_interval,
            blit=True,
            cache_frame_data=False,
        )

        plt.show()

    def close(self):
        """关闭可视化窗口"""
        plt.close(self.fig)


class TimeSeriesPlotter:
    """用于可视化随时间变化的数据的类"""

    def __init__(
        self,
        max_points: int = 500,
        title: str = "数据实时显示",
        xlabel: str = "时间 (秒)",
        ylabel: str = "数值",
        y_range: Optional[Tuple[float, float]] = None,
    ):
        """
        初始化时序数据绘图器

        参数:
            max_points: 最大数据点数
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            y_range: y轴范围(最小值,最大值),None表示自动调整
        """
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)
        self.start_time = time.time()

        # 创建图表
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()
        (self.line,) = self.ax.plot([], [], "b-", label="数据")

        # 设置图表属性
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.ax.legend()

        # 设置初始显示范围
        self.ax.set_xlim(0, 10)
        if y_range:
            self.ax.set_ylim(*y_range)
        else:
            self.ax.set_ylim(0, 1)

        # 添加FPS显示
        self.fps_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)
        self.last_time = time.time()
        self.frame_count = 0

        # 显示图表
        plt.show(block=False)
        plt.pause(0.1)

    def add_data(self, value: float, close_callback: Optional[Callable] = None):
        """
        添加新的数据点

        参数:
            value: 数据值
            close_callback: 窗口关闭时的回调函数
        """
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        self.values.append(value)

        # 更新FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - self.last_time)
            self.fps_text.set_text(f"FPS: {fps:.1f}")
            self.last_time = current_time

        # 更新数据
        self.line.set_data(list(self.times), list(self.values))

        # 自动调整显示范围
        if len(self.times) > 0:
            xmin, xmax = min(self.times), max(self.times)
            ymin, ymax = min(self.values), max(self.values)
            margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
            self.ax.set_xlim(xmin, xmax + 1)
            self.ax.set_ylim(ymin - margin, ymax + margin)

        # 刷新图表
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # 设置关闭事件处理
        if close_callback and not hasattr(self, "_close_callback_set"):
            self.fig.canvas.mpl_connect("close_event", lambda event: close_callback())
            self._close_callback_set = True

    def close(self):
        """关闭绘图窗口"""
        plt.close(self.fig)
