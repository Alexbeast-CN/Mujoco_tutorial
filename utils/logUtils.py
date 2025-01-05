from loguru import logger
import sys
from pathlib import Path
from datetime import datetime
import time
from typing import List, Dict


class RateLimitedLogger:
    """带速率限制的Logger包装器"""

    def __init__(self, name: str, logger_manager: "LoggerManager"):
        self.name = name
        self.logger_manager = logger_manager
        self._logger = logger.bind(name=name)

    @property
    def is_muted(self) -> bool:
        """检查logger是否被屏蔽"""
        return self.name in self.logger_manager.muted_loggers

    def debug(self, message: str):
        if not self.is_muted:
            self._logger.debug(message)

    def info(self, message: str):
        if not self.is_muted:
            self._logger.info(message)

    def warning(self, message: str):
        if not self.is_muted:
            self._logger.warning(message)

    def error(self, message: str):
        if not self.is_muted:
            self._logger.error(message)

    def debug_rate_limited(
        self, message: str, interval: float = 0.5, pattern_key: str = None
    ):
        if not self.is_muted:
            self.logger_manager.log_with_rate_limit(
                self.name, message, interval, pattern_key, level="DEBUG"
            )

    def info_rate_limited(
        self, message: str, interval: float = 0.5, pattern_key: str = None
    ):
        if not self.is_muted:
            self.logger_manager.log_with_rate_limit(
                self.name, message, interval, pattern_key, level="INFO"
            )

    def warning_rate_limited(
        self, message: str, interval: float = 0.5, pattern_key: str = None
    ):
        if not self.is_muted:
            self.logger_manager.log_with_rate_limit(
                self.name, message, interval, pattern_key, level="WARNING"
            )

    def error_rate_limited(
        self, message: str, interval: float = 0.5, pattern_key: str = None
    ):
        if not self.is_muted:
            self.logger_manager.log_with_rate_limit(
                self.name, message, interval, pattern_key, level="ERROR"
            )


class LoggerManager:
    """日志管理器"""

    def __init__(self):
        logger.remove()
        self._loggers: Dict[str, RateLimitedLogger] = {}
        self._rate_limit_times = {}
        self._muted_loggers = set()  # 存储被屏蔽的logger名称

    @property
    def muted_loggers(self) -> set:
        """获取被屏蔽的logger��称集合"""
        return self._muted_loggers

    def mute(self, *logger_names: str) -> None:
        """
        屏蔽指定的logger

        Args:
            *logger_names: 要屏蔽的logger名称
        """
        self._muted_loggers.update(logger_names)

    def unmute(self, *logger_names: str) -> None:
        """
        取消屏蔽指定的logger

        Args:
            *logger_names: 要取消屏蔽的logger名称
        """
        self._muted_loggers.difference_update(logger_names)

    def unmute_all(self) -> None:
        """取消所有logger的屏蔽"""
        self._muted_loggers.clear()

    def setup(
        self,
        enable_file_output: bool = True,
        log_dir: str = "logs",
        log_file_prefix: str = "",
        console_format: str = None,
        file_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {extra[name]} | {level} | {message}",
        log_level: str = "DEBUG",
        context_names: List[str] = None,
    ) -> None:
        """设置日志系统"""
        # 设置默认的彩色控制台格式
        if console_format is None:
            console_format = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<cyan>{extra[name]}</cyan> | "
                "<level>{level}</level> | "
                "<white>{message}</white>"
            )

        # 修改 filter 逻辑:如果没有指定 context_names,允许所有 logger 输出
        filter_func = (
            (lambda record: record["extra"].get("name", "") in context_names)
            if context_names
            else None
        )

        # 添加控制台输出
        logger.add(
            sys.stdout,
            format=console_format,
            filter=filter_func,
            level=log_level,
            colorize=True,
        )

        # 只在enable_file_output为True时添加文件输出
        if enable_file_output:
            # 创建日志目录(包括所有父目录)
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # 生成日志文件名
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if log_file_prefix:
                log_file = log_dir / f"{log_file_prefix}_{timestamp}.log"
            else:
                log_file = log_dir / f"app_{timestamp}.log"

            # 添加文件输出
            logger.add(
                str(log_file),
                format=file_format,
                encoding="utf-8",
                rotation="100 MB",
                level=log_level,
                colorize=True,
            )

        # 创建上下文loggers
        if context_names:
            for name in context_names:
                self._loggers[name] = RateLimitedLogger(name, self)

    def get_logger(self, name: str) -> RateLimitedLogger:
        """获取带速率限制功能的logger"""
        if name not in self._loggers:
            self._loggers[name] = RateLimitedLogger(name, self)
        return self._loggers[name]

    def _extract_pattern_key(self, message: str) -> str:
        """
        从消息中提取模式标识符

        Args:
            message: 日志消息

        Returns:
            提取出的模式标识符
        """
        # 分割消息，处理常见的格式
        parts = message.split(":")
        if len(parts) > 1:
            # 如果消息格式是 "固定部分: 变化部分"，使用冒号前的部分
            return parts[0].strip()

        # 查找并移除常见的变化部分（数字、时间戳等）
        import re

        # 移除数字（包括小数）
        pattern = re.sub(r"\d+\.?\d*", "", message)
        # 移除时间戳格式
        pattern = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "", pattern)
        # 移除多余的空格
        pattern = re.sub(r"\s+", " ", pattern).strip()

        return pattern if pattern else message

    def log_with_rate_limit(
        self,
        name: str,
        message: str,
        interval: float,
        pattern_key: str = None,
        level: str = "INFO",
    ):
        """
        记录带有速率限制的日志，自动提取或使用提供的pattern_key

        Args:
            name: logger的名称
            message: 要记录的消息
            interval: 最小间隔时间(秒)
            pattern_key: 可选的消息模式标识符。如果不提供，将自动从消息中提取
            level: 日志级别，默认为"INFO"
        """

        # 检查是否被屏蔽
        if name in self._muted_loggers:
            return

        if name not in self._loggers:
            self._loggers[name] = RateLimitedLogger(name, self)

        current_time = time.time()
        # 如果没有提供pattern_key，则自动提取
        if pattern_key is None:
            pattern_key = self._extract_pattern_key(message)

        rate_limit_key = f"{name}_{pattern_key}_last_time"

        if not hasattr(self, "_rate_limit_times"):
            self._rate_limit_times = {}

        last_time = self._rate_limit_times.get(rate_limit_key, 0)

        if current_time - last_time >= interval:
            # 根据level选择对应的日志方法
            logger_method = getattr(self._loggers[name]._logger, level.lower())
            logger_method(message)
            self._rate_limit_times[rate_limit_key] = current_time


# 创建全局logger管理器实例
logger_manager = LoggerManager()


def setup_logger(
    enable_file_output: bool = True,
    log_dir: str = "logs",
    log_file_prefix: str = "",
    context_names: List[str] = None,
    log_level: str = "DEBUG",
) -> LoggerManager:
    """
    快速设置日志系统

    Args:
        enable_file_output: 是否启用文件输出,默认为 True
        log_dir: 日志目录
        log_file_prefix: 日志文件名前缀
        context_names: 日志上下文名称列表
        log_level: 日志级别

    Returns:
        LoggerManager实例
    """
    logger_manager.setup(
        enable_file_output=enable_file_output,
        log_dir=log_dir,
        log_file_prefix=log_file_prefix,
        context_names=context_names,
        log_level=log_level,
    )
    return logger_manager


if __name__ == "__main__":
    # 设置日志系统
    setup_logger(context_names=["example"])
    i = 0
    example_logger = logger_manager.get_logger("example")
    while i < 20:
        # 测试带有速率限制的日志记录
        # 不需要显式提供pattern_key，会自动从消息中提取
        example_logger.info_rate_limited(f"消息1: {i}", 0.5)

        example_logger.info_rate_limited(f"消息2: {i}", 1.0)

        example_logger.info_rate_limited(f"温度值为 {23.5} 度", 2.0)

        example_logger.info_rate_limited(f"当前时间 2024-01-01 12:00:00 处理完成", 1.0)

        time.sleep(0.1)
        i += 1
