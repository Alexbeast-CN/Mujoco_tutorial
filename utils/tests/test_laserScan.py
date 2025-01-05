import pytest
import numpy as np
import mujoco
import time
from utils.sensorUtils import LaserScan, LineLaser, ThreadedSensor
from utils.logUtils import setup_logger

# 将XML_STRING移到顶层
XML_STRING = """
<mujoco model="laser_scanner">
    <visual>
        <scale forcewidth="0.01" contactwidth="0.05" contactheight="0.05"/>
    </visual>
    
    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" castshadow="false"/>
        
        <!-- 四面墙 -->
        <geom name="wall1" type="box" size="0.5 0.01 0.2" pos="0 0.5 0.2"/>  
        <geom name="wall2" type="box" size="0.5 0.01 0.2" pos="0 -0.5 0.2"/>
        <geom name="wall3" type="box" size="0.01 0.5 0.2" pos="0.5 0 0.2"/>
        <geom name="wall4" type="box" size="0.01 0.5 0.2" pos="-0.5 0 0.2"/>
        
        <!-- 传感器安装位置 -->
        <body name="sensors" pos="0 0 0.1">
            <geom name="ball" type="sphere" size="0.05" rgba="1 0 0 1"/>
            <joint name="sensor_x" type="slide" axis="1 0 0"/>
            <joint name="sensor_y" type="slide" axis="0 1 0"/>
            <joint name="sensor_z" type="slide" axis="0 0 1"/>
            <!-- 基本线激光阵列 -->
            <replicate count="10" offset="0.02 0 0">
                <site name="line_laser" pos="0 0 0" zaxis="0 1 0" rgba="1 0 0 1" size="0.01"/>
            </replicate>
        </body>
    </worldbody>
    
    <sensor>
        <!-- 基本线激光测试用传感器 -->
        <rangefinder name="line_laser" site="line_laser"/>
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
        log_file_prefix="test_laser_scan", context_names=["test_laser_scan"]
    )
    return logger_manager.get_logger("test_laser_scan")


@pytest.fixture(scope="module")
def mj_model():
    """创建包含多个传感器的MuJoCo模型"""
    try:
        return mujoco.MjModel.from_xml_string(XML_STRING)
    except Exception as e:
        pytest.skip(f"无法加载MuJoCo模型: {str(e)}")


@pytest.fixture
def mj_data(mj_model):
    """创建并重置MuJoCo数据"""
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)
    return data


@pytest.fixture
def basic_scanner(mj_model, mj_data, sensor_logger):
    """创建基本的激光扫描器"""
    return LaserScan(
        model=mj_model,
        data=mj_data,
        frequency=20,
        logger=sensor_logger,
        sensor_prefix="line_laser",
        site_prefix="line_laser",
    )


@pytest.fixture
def line_scanner(mj_model, mj_data, sensor_logger):
    """创建线激光扫描器"""
    return LineLaser(
        model=mj_model,
        data=mj_data,
        frequency=20,
        logger=sensor_logger,
        sensor_prefix="line_laser",
        site_prefix="line_laser",
    )


class TestLaserScan:
    """LaserScan类测试"""

    def test_init_basic(self, basic_scanner):
        """测试基本初始化"""
        assert basic_scanner.sensor_count > 0
        assert len(basic_scanner.sensors) == basic_scanner.sensor_count
        assert all(sensor is not None for sensor in basic_scanner.sensors)

    def test_sensor_naming(self, basic_scanner):
        """测试传感器命名"""
        # 检查传感器命名
        assert all(name.startswith("line_laser") for name in basic_scanner.sensor_names)
        assert all(name.startswith("line_laser") for name in basic_scanner.site_names)
        assert len(basic_scanner.sensor_names) == basic_scanner.sensor_count
        assert len(basic_scanner.site_names) == basic_scanner.sensor_count

    def test_reading_frame(self, basic_scanner):
        """测试帧读取"""
        # 读取一帧数据
        frame = basic_scanner.reading

        # 验证帧数据
        assert frame is not None
        assert len(frame.points) == basic_scanner.sensor_count
        assert frame.timestamp > 0

        # 检查点数据
        for point in frame.points:
            assert 0 <= point.distance <= 2.0  # 在cutoff范围之内
            assert point.valid
            assert isinstance(point.position, np.ndarray)
            assert isinstance(point.direction, np.ndarray)

    def test_threaded_scanning(self, basic_scanner):
        """测试线程化扫描"""
        threaded = ThreadedSensor(basic_scanner)

        try:
            # 启动描
            threaded.start()
            time.sleep(0.1)  # 等待数据采集

            # 获取最新帧
            frame = threaded.reading
            assert frame is not None
            assert len(frame.points) == basic_scanner.sensor_count

        finally:
            threaded.stop()

    def test_recording(self, basic_scanner, tmp_path):
        """测试数据记录功能"""
        # 开始记录
        assert basic_scanner.recording_on(str(tmp_path))

        try:
            # 记录几帧数据
            for _ in range(5):
                basic_scanner.reading
                time.sleep(0.02)

            # 停止记录
            basic_scanner.recording_off()

            # 验证文件
            csv_files = list(tmp_path.glob("*.csv"))
            assert len(csv_files) == 1
            assert csv_files[0].stat().st_size > 0

            # 验证文件内容
            content = csv_files[0].read_text()
            header = content.splitlines()[0]
            assert "timestamp" in header
            assert len(header.split(",")) == basic_scanner.sensor_count * 4 + 1

            line1 = content.splitlines()[1].split(",")
            assert len(line1) == basic_scanner.sensor_count * 4 + 1
            assert float(line1[1]) == pytest.approx(0.49, abs=0.1)

            assert len(content) > 5

        finally:
            basic_scanner.recording_off()

    def test_threaded_recording(self, mj_model, mj_data, basic_scanner, tmp_path):
        """测试线程化记录"""
        start_time = time.time()
        frequency = basic_scanner.frequency
        step_size = 0.001
        test_time = 0.4

        threaded = ThreadedSensor(basic_scanner)
        assert threaded.recording_on(str(tmp_path))

        # 记录数据
        readings = []
        print(mj_data.qpos)
        while time.time() - start_time < test_time:
            mj_data.qpos[1] = mj_data.qpos[1] + step_size
            mujoco.mj_step(mj_model, mj_data)
            current_frame = threaded.reading
            if current_frame is not None and current_frame.distances is not []:
                if len(readings) == 0 or current_frame != readings[-1]:
                    readings.append(current_frame)
            time.sleep(1 / frequency / 2)

        # 停止记录
        threaded.recording_off()
        threaded.stop()

        # 验证文件
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1
        assert csv_files[0].stat().st_size > 0

        # 验证文件内容
        content = csv_files[0].read_text()
        lines = content.splitlines()
        assert len(lines) == len(readings) + 1

        # 检查表头
        header = lines[0]
        assert "timestamp" in header
        value_count = basic_scanner.sensor_count * 4
        assert len(header.split(",")) == value_count + 1

        for i in range(len(readings)):
            assert float(lines[i + 1].split(",")[0]) == pytest.approx(
                i * 1 / frequency, abs=0.01
            )
            for j in range(1, basic_scanner.sensor_count + 1):
                assert float(lines[i + 1].split(",")[j]) == pytest.approx(
                    readings[i].distances[j - 1], abs=0.01
                )
                assert float(lines[i + 1].split(",")[j + 1]) == pytest.approx(
                    readings[i].positions[j - 1][0], abs=0.01
                )
                assert float(lines[i + 1].split(",")[j + 2]) == pytest.approx(
                    readings[i].positions[j - 1][1], abs=0.01
                )
                assert float(lines[i + 1].split(",")[j + 3]) == pytest.approx(
                    readings[i].positions[j - 1][2], abs=0.01
                )


class TestLineLaser:
    """LineLaser类测试"""

    def test_init_validation(self, mj_model, mj_data, line_scanner):
        """测试初始化和验证"""
        mujoco.mj_forward(mj_model, mj_data)

        # 测试平面法向量
        assert line_scanner.plane_normal is not None
        assert isinstance(line_scanner.plane_normal, np.ndarray)

        # 测试变换矩阵
        assert line_scanner.transform_matrix is not None
        assert line_scanner.transform_matrix.shape == (3, 3)
        assert np.allclose(
            np.dot(line_scanner.transform_matrix, line_scanner.transform_matrix.T),
            np.eye(3),
        )

        # 测试原点
        assert line_scanner.origin is not None
        assert isinstance(line_scanner.origin, np.ndarray)
        assert line_scanner.origin.shape == (3,)

    def test_coordinate_transform(self, line_scanner):
        """测试坐标变换"""
        assert np.allclose(line_scanner.origin, np.array([0, 0, 0.1]), atol=0.001)
        assert np.allclose(line_scanner.plane_normal, np.array([0, 0, 1.0]))
        # 变换矩阵不是很好确定

    def test_line_laser_frame(self, line_scanner):
        """测试线激光帧"""
        # 读取一帧
        frame = line_scanner.reading

        # 验证帧属性
        assert frame.plane_normal is not None
        assert np.allclose(frame.plane_normal, np.array([0, 0, 1.0]))
        assert frame.origin is not None
        assert np.allclose(frame.origin, np.array([0, 0, 0.1]))
        assert frame.transform_matrix is not None

        # 验证2D位置有效性
        positions_2d = frame.positions_2d
        assert positions_2d is not None
        assert len(positions_2d) == line_scanner.sensor_count
        assert np.allclose(positions_2d[0], np.array([0.0, 0.49]))
        assert np.allclose(positions_2d[9], np.array([0.18, 0.49]))

        # 测试3D转换
        point_3d0 = frame.transform_to_3d(positions_2d[0])
        point_3d9 = frame.transform_to_3d(positions_2d[9])
        outside_point = frame.transform_to_3d(np.array([0.2, 0.2]))
        assert point_3d0 is not None
        assert point_3d0.shape == (3,)
        assert np.allclose(point_3d0, np.array([0.0, 0.49, 0.1]))
        assert np.allclose(point_3d9, np.array([0.18, 0.49, 0.1]))
        assert np.allclose(outside_point, np.array([0.2, 0.2, 0.1]))

    def test_batch_transform(self, line_scanner):
        """测试批量坐标转换"""
        # 创建测试数据
        points_2d = np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

        # 获取一帧用于转换
        frame = line_scanner.reading

        # 批量转换到3D
        points_3d = frame.transform_points_to_3d(points_2d)
        assert points_3d is not None
        assert points_3d.shape == (3, 3)
        assert np.all(np.isfinite(points_3d))

        # 验证转换一致性
        for i in range(len(points_2d)):
            single_3d = frame.transform_to_3d(points_2d[i])
            assert np.allclose(single_3d, points_3d[i])


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
