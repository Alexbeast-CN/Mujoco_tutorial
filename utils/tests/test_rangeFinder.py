import pytest
import numpy as np
import mujoco
import time
from utils.sensorUtils import RangeFinder, ThreadedSensor
from utils.logUtils import setup_logger

# 将XML_STRING移到顶层
XML_STRING = """
<mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
        <!-- 参考物体 -->
        <body name="target" pos="0 0 1">
            <geom type="box" size="0.1 0.1 0.1"/>
        </body>
        
        <!-- 传感器安装位置 -->
        <body name="sensor" pos="0 0 0">
            <geom name="ball" type="sphere" size="0.05" rgba="1 0 0 1"/>
            <joint name="sensor_x" type="slide" axis="1 0 0"/>
            <joint name="sensor_y" type="slide" axis="0 1 0"/>
            <joint name="sensor_z" type="slide" axis="0 0 1"/>
            <site name="sensor_site" pos="0 0 0" euler="0 0 0"/>
        </body>
    </worldbody>
    
    <sensor>
        <!-- 距离传感器 -->
        <rangefinder name="test_sensor" site="sensor_site" cutoff="2"/>
        <!-- 带噪声的传感器 -->
        <rangefinder name="noisy_sensor" site="sensor_site" cutoff="2" noise="0.01"/>
    </sensor>
    
    <actuator>
        <motor joint="sensor_x" name="ax" gear="100"/>
        <motor joint="sensor_y" name="ay" gear="100"/>
        <motor joint="sensor_z" name="az" gear="100"/>
    </actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def sensor_logger():
    """创建测试用logger"""
    logger_manager = setup_logger(
        log_file_prefix="test_rangefinder", context_names=["test_rangefinder"]
    )
    logger = logger_manager.get_logger("test_rangefinder")
    return logger


@pytest.fixture(scope="module")
def mj_model():
    """创建MuJoCo模型"""
    model = mujoco.MjModel.from_xml_string(XML_STRING)
    return model


@pytest.fixture
def mj_data(mj_model):
    """创建并重置MuJoCo数据"""
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture
def basic_sensor(mj_model, mj_data, sensor_logger):
    """创建基本的传感器对象"""
    return RangeFinder(
        mj_model,
        mj_data,
        "test_sensor",
        "sensor_site",
        frequency=50,
        logger=sensor_logger,
    )


@pytest.fixture
def noisy_sensor(mj_model, mj_data, sensor_logger):
    """创建带噪声的传感器对象"""
    return RangeFinder(
        mj_model,
        mj_data,
        "noisy_sensor",
        "sensor_site",
        frequency=50,
        logger=sensor_logger,
    )


def test_constructor_valid(basic_sensor):
    """测试有效的构造参数"""
    assert basic_sensor is not None


@pytest.mark.parametrize(
    "sensor_name,site_name,error",
    [
        ("invalid_sensor", "sensor_site", ValueError),
        ("test_sensor", "invalid_site", ValueError),
    ],
)
def test_constructor_invalid(
    mj_model, mj_data, sensor_logger, sensor_name, site_name, error
):
    """测试无效的构造参数"""
    with pytest.raises(error):
        RangeFinder(mj_model, mj_data, sensor_name, site_name, logger=sensor_logger)


def test_basic_reading(basic_sensor):
    """测试基本的读数功能"""
    reading = basic_sensor.reading
    assert reading == pytest.approx(0.9, rel=1e-3)


def test_noise_statistics(mj_model, mj_data, noisy_sensor):
    """测试传感器噪声的统计特性"""
    readings = []
    for _ in range(100):
        # 重新前向计算,让噪声生效
        mujoco.mj_forward(mj_model, mj_data)
        readings.append(noisy_sensor.reading)
        # 物理仿真步进
        mujoco.mj_step(mj_model, mj_data)

    readings = np.array(readings)

    # 均值应该在0.9附近(考虑盒子宽度0.1)
    assert np.mean(readings) == pytest.approx(0.9, abs=0.1)

    # 标准差应该接近设定的噪声值0.01
    assert np.std(readings) == pytest.approx(0.01, abs=0.1)


def test_moving_reading(mj_model, mj_data, noisy_sensor):
    """测试在传感器更换位置后读数的变化"""
    expected_distance = 0.9
    step_size = 0.1
    for _ in range(5):
        mj_data.qpos[2] = mj_data.qpos[2] + step_size
        expected_distance = expected_distance - step_size
        mujoco.mj_step(mj_model, mj_data)
        assert noisy_sensor.reading == pytest.approx(expected_distance, abs=0.03)

    # 检查传感器是否能正确处理无效的读数
    mj_data.qpos[0] = mj_data.qpos[0] + 1
    mujoco.mj_step(mj_model, mj_data)
    assert noisy_sensor.reading == pytest.approx(-1, abs=0.03)


def test_recording(mj_model, mj_data, basic_sensor, tmp_path):
    """测试数据记录功能"""
    # 开始记录
    assert basic_sensor.recording_on(str(tmp_path))
    time_step = 0.01
    readings = []

    # 记录一些数据
    for _ in range(10):
        # 物理仿真步进
        mujoco.mj_step(mj_model, mj_data)
        readings.append(basic_sensor.reading)
        time.sleep(time_step)

    # 停止记录
    basic_sensor.recording_off()

    # 验证文件
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].stat().st_size > 0

    # 验证文件内容
    content = csv_files[0].read_text()
    lines = content.splitlines()
    assert len(lines) > 10

    # 检查表头
    assert "timestamp,value" in lines[0]
    for i in range(10):
        assert float(lines[i + 1].split(",")[0]) == pytest.approx(
            i * time_step, abs=0.01
        )
        assert float(lines[i + 1].split(",")[1]) == pytest.approx(readings[i], abs=0.01)


def test_measurement_info(basic_sensor):
    """测试测量信息获取"""
    info = basic_sensor.get_measurement_info()

    assert info is not None
    assert info.distance == pytest.approx(0.9, abs=1e-2)
    assert info.valid

    # 检查位置和方向
    np.testing.assert_array_almost_equal(info.position, np.array([0, 0, 0.9]))
    np.testing.assert_array_almost_equal(info.site, np.array([0, 0, 0.0]))
    np.testing.assert_array_almost_equal(info.direction, np.array([0, 0, 1]))


def test_basic_threaded_sensor(basic_sensor):
    """测试线程化传感器"""
    threaded = ThreadedSensor(basic_sensor)

    # 启动传感器
    threaded.start()
    try:
        # 等待数据采集
        time.sleep(0.1)
        assert threaded.reading == pytest.approx(0.9, abs=1e-2)
    finally:
        # 确保停止传感器
        threaded.stop()


def test_threaded_sensor_recording(mj_model, mj_data, basic_sensor, tmp_path):
    """测试线程化传感器的录制功能"""
    start_time = time.time()
    frequency = basic_sensor.frequency
    step_size = 0.001
    test_time = 0.4

    threaded = ThreadedSensor(basic_sensor)
    assert threaded.recording_on(str(tmp_path))

    # 记录数据
    readings = []
    while time.time() - start_time < test_time:
        mj_data.qpos[2] = mj_data.qpos[2] + step_size
        mujoco.mj_step(mj_model, mj_data)

        current_value = threaded.reading
        if current_value is not None:
            if len(readings) == 0 or current_value != readings[-1]:
                readings.append(current_value)
        time.sleep(1 / frequency / 2)

    # 停止记录
    assert threaded.recording_off()
    threaded.stop()

    # 验证文件
    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) == 1
    assert csv_files[0].stat().st_size > 0

    # 验证文件内容
    content = csv_files[0].read_text()
    lines = content.splitlines()
    assert len(lines) > len(readings)

    # 检查表头
    assert "timestamp,value" in lines[0]
    for i in range(len(readings)):
        assert float(lines[i + 1].split(",")[0]) == pytest.approx(
            i * 1 / frequency, abs=0.01
        )
        assert float(lines[i + 1].split(",")[1]) == pytest.approx(readings[i], abs=0.01)


def test_threaded_sensor_frequency(mj_model, mj_data, basic_sensor):
    """测试线程化传感器的频率"""

    # 创建线程化传感器
    threaded = ThreadedSensor(basic_sensor)
    # 获取传感器的频率
    frequency = basic_sensor.frequency
    # 启动传感器
    threaded.start()

    # 设置一个很小的步长,确保传感器能准确读取
    step_size = 0.00001

    # 设置一个较长的测试时间,确保传感器能准确读取
    test_time = 0.4

    # 记录开始时间
    start_time = time.time()
    # 记录读数
    readings = []
    average_frame_rate = 0

    while time.time() - start_time < test_time:
        mj_data.qpos[2] = mj_data.qpos[2] + step_size
        mujoco.mj_step(mj_model, mj_data)
        current_value = threaded.reading
        current_frame_rate = threaded.get_frame_rate()
        average_frame_rate = (average_frame_rate + current_frame_rate) / 2
        if current_value is not None:
            if len(readings) == 0 or current_value != readings[-1]:
                readings.append(current_value)
        time.sleep(1 / frequency / 2)

    threaded.stop()
    assert len(readings) == pytest.approx(frequency * test_time, rel=0.3)
    assert average_frame_rate == pytest.approx(frequency, rel=0.3)


def test_moving_threaded_sensor(mj_model, mj_data, basic_sensor):
    """测试移动中的线程化传感器"""
    threaded = ThreadedSensor(basic_sensor)
    threaded.start()
    try:
        expected_distance = 0.9
        step_size = 0.1
        for _ in range(5):
            mj_data.qpos[2] = mj_data.qpos[2] + step_size
            expected_distance = expected_distance - step_size
            mujoco.mj_step(mj_model, mj_data)
            time.sleep(basic_sensor.frequency / 1000 * 4)
            assert threaded.reading == pytest.approx(expected_distance, abs=0.1)
    finally:
        threaded.stop()


def test_concurrent_sensors(mj_model, mj_data, sensor_logger):
    """测试多传感器并行工作"""
    sensors = [
        ThreadedSensor(
            RangeFinder(
                mj_model,
                mj_data,
                "test_sensor",
                "sensor_site",
                frequency=20,
                logger=sensor_logger,
            )
        ),
        ThreadedSensor(
            RangeFinder(
                mj_model,
                mj_data,
                "noisy_sensor",
                "sensor_site",
                frequency=10,
                logger=sensor_logger,
            )
        ),
    ]

    # 启动所有传感器
    for sensor in sensors:
        sensor.start()

    try:
        # 等待数据采集
        time.sleep(0.1)

        # 验证所有传感器都能读取数据
        for sensor in sensors:
            assert sensor.reading is not None
            assert 0.8 <= sensor.reading <= 1.0  # 考虑噪声的范围检查

    finally:
        # 停止所有传感器
        for sensor in sensors:
            sensor.stop()


def test_recording_errors(basic_sensor, tmp_path):
    """测试记录功能的错误处理"""
    # 重复开始记录
    basic_sensor.recording_on(str(tmp_path))
    assert not basic_sensor.recording_on(str(tmp_path))

    # 停止未开始的记录
    basic_sensor.recording_off()
    basic_sensor.recording_off()  # 应该不会报错


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # 将项目根目录添加到Python路径
    ROOT_DIR = Path(__file__).parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    import mujoco.viewer

    # 使用相同的XML_STRING创建model和data
    model = mujoco.MjModel.from_xml_string(XML_STRING)
    data = mujoco.MjData(model)

    # 创建viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 渲染循环
        while viewer.is_running():
            mujoco.mj_step(model, data)

            # 更新viewer
            viewer.sync()

            # 控制渲染频率
            time.sleep(0.01)
