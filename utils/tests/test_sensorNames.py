import pytest
import mujoco
import numpy as np
from utils.sensorUtils import SensorNameGenerator


class TestSensorNameGenerator:
    @pytest.fixture
    def mj_model(self):
        """加载测试用的 MuJoCo 模型"""
        XML_STRING = """
            <mujoco>
                <option timestep="0.005" integrator="Euler" gravity="0 0 0"/>
                
                <worldbody>            
                    <body name="sensor_body" pos="0 0 0">
                        <geom type="box" size="0.1 0.1 0.1"/>
                        
                        <!-- 基本序号命名测试 -->
                        <replicate count="5" offset="0.001 0 0">
                            <site name="sensor" pos="0 0 0" zaxis="0 0 -1" rgba="1 0 0 0"/>
                        </replicate>
                        
                        <!-- 带分隔符命名测试 -->
                        <replicate count="10" offset="0.001 0 0">
                            <site name="line_laser" pos="-0.02 0 0.025" zaxis="0 0 -1" 
                                    rgba="1 0 0 0"/>
                        </replicate>
                        
                        <!-- 大数量命名测试 -->
                        <replicate count="80" offset="0.0005 0 0">
                            <site name="laser_array" pos="-0.03 0 0.025" zaxis="0 0 -1" 
                                    rgba="1 0 0 0"/>
                        </replicate>
                    </body>
                </worldbody>
                
                <sensor>
                    <!-- 基本序号命名测试 -->
                    <rangefinder name="sensor" site="sensor" cutoff="0.08"/>
                    
                    <!-- 带分隔符命名测试 -->
                    <rangefinder name="line_laser" site="line_laser" cutoff="0.08"/>
                    
                    <!-- 大数量命名测试 -->
                    <rangefinder name="laser_array" site="laser_array" cutoff="0.08"/>
                </sensor>
            </mujoco>
            """
        try:
            return mujoco.MjModel.from_xml_string(XML_STRING)
        except Exception as e:
            pytest.skip(f"无法加载 MuJoCo 模型: {str(e)}")

    def test_basic_naming_matches_mujoco(self, mj_model):
        """测试基本的序号命名是否与 MuJoCo 匹配"""
        # 从 MuJoCo 获取传感器名称
        mj_names = []
        for i in range(5):  # 获取前5个传感器
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                mj_names.append(name)

        # 使用 SensorNameGenerator 生成名称
        gen_names = SensorNameGenerator.generate_names("sensor", 5)

        assert gen_names == mj_names, "生成的名称与 MuJoCo 不匹配"

    def test_separator_naming_matches_mujoco(self, mj_model):
        """测试带分隔符的命名是否与 MuJoCo 匹配"""
        # 从 MuJoCo 获取传感器名称
        mj_names = []
        for i in range(5, 15):  # 获取第6-15个传感器
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                mj_names.append(name)

        # 使用 SensorNameGenerator 生成名称
        gen_names = SensorNameGenerator.generate_names("line_laser", 10)

        assert gen_names == mj_names, "带分隔符的名称与 MuJoCo 不匹配"

    def test_large_count_naming_matches_mujoco(self, mj_model):
        """测试大数量命名是否与 MuJoCo 匹配"""
        # 从 MuJoCo 获取传感器名称
        mj_names = []
        for i in range(15, 95):  # 获取第16-95个传感器
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                mj_names.append(name)

        # 使用 SensorNameGenerator 生成名称
        gen_names = SensorNameGenerator.generate_names("laser_array", 80)

        assert gen_names == mj_names, "大数量命名与 MuJoCo 不匹配"

    def test_validate_count(self):
        """测试validate_count方法"""
        # 正常情况
        SensorNameGenerator.validate_count(0)
        SensorNameGenerator.validate_count(1)
        SensorNameGenerator.validate_count(100)

        # 错误情况
        with pytest.raises(ValueError, match="传感器数量不能为负数"):
            SensorNameGenerator.validate_count(-1)

        with pytest.raises(ValueError, match="传感器数量不能为负数"):
            SensorNameGenerator.validate_count(-100)

    def test_validate_prefix(self):
        """测试validate_prefix方法"""
        # 正常情况
        valid_prefixes = ["sensor", "Sensor_1", "laser_array", "SENSOR_123"]
        for prefix in valid_prefixes:
            SensorNameGenerator.validate_prefix(prefix)

        # 错误情况
        invalid_prefixes = [
            "",  # 空字符串
            " ",  # 空格
            "sensor@",  # 特殊字符
            "laser#123",
            "sensor space",  # 包含空格
        ]
        for prefix in invalid_prefixes:
            with pytest.raises(ValueError):
                SensorNameGenerator.validate_prefix(prefix)

    def test_generate_range_names(self):
        """测试generate_range_names方法"""
        # 基本功能测试
        names = SensorNameGenerator.generate_range_names("sensor", 2, 5)
        assert len(names) == 3
        assert names == ["sensor2", "sensor3", "sensor4"]

        # 测试分隔符
        names = SensorNameGenerator.generate_range_names("laser", 0, 3, separator="_")
        assert names == ["laser_0", "laser_1", "laser_2"]

        # 测试位数自动调整
        names = SensorNameGenerator.generate_range_names("dev", 8, 12)
        assert len(names) == 4
        assert names[0] == "dev08"
        assert names[-1] == "dev11"

        # 错误情况测试
        with pytest.raises(ValueError, match="起始编号必须小于结束编号"):
            # 起始等于结束
            SensorNameGenerator.generate_range_names("sensor", 5, 5)

        with pytest.raises(ValueError, match="起始编号必须小于结束编号"):
            # 起始大于结束
            SensorNameGenerator.generate_range_names("sensor", 6, 5)

        with pytest.raises(ValueError):
            # 无效前缀
            SensorNameGenerator.generate_range_names("", 0, 5)

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试单个传感器
        names = SensorNameGenerator.generate_names("sensor", 1)
        assert names == ["sensor0"]

        # 测试零个传感器
        names = SensorNameGenerator.generate_names("sensor", 0)
        assert names == []

        # 测试大数量
        count = 1000
        names = SensorNameGenerator.generate_names("sensor", count)
        assert len(names) == count
        assert names[0] == "sensor0000"
        assert names[-1] == f"sensor{count-1:04d}"

        # 测试相邻范围
        names = SensorNameGenerator.generate_range_names("sensor", 1, 2)
        assert names == ["sensor1"]
