import mujoco
import numpy as np
from typing import List, Optional
import yaml

class JointInfo:
    def __init__(self, model, joint_names: List[str], config_file: Optional[str] = None):
        """初始化关节信息
        
        Args:
            model: MuJoCo 模型
            joint_names: 关节名称列表
            config_file: 配置文件路径，包含速度和加速度限制
        """
        self.model = model
        self.joint_names = joint_names
        self.joint_ranges = []
        self.joint_qpos_ids = []
        self.joint_qvel_ids = []
        
        # 默认限制
        self.velocity_limits = None
        self.acceleration_limits = None
        self.jerk_limits = None
        
        self._init_joint_info()
        
        # 如果提供了配置文件，则加载限制
        if config_file:
            self.load_limits_from_config(config_file)
            
    def load_limits_from_config(self, config_file: str):
        """从配置文件加载运动限制
        
        Args:
            config_file: YAML配置文件路径
        """
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        self.velocity_limits = config.get('max_velocity')
        self.acceleration_limits = config.get('max_acceleration')
        self.jerk_limits = config.get('max_jerk')
        
    @property
    def dof(self) -> int:
        """获取自由度数量"""
        return len(self.joint_names)
        
    def _init_joint_info(self):
        """获取关节信息"""
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ranges.append(self.model.jnt_range[joint_id])
            self.joint_qpos_ids.append(self.model.jnt_qposadr[joint_id])
            self.joint_qvel_ids.append(self.model.jnt_dofadr[joint_id])
            
    def get_random_joint_positions(self):
        """生成随机关节位置"""
        return [np.random.uniform(joint_range[0], joint_range[1]) 
                for joint_range in self.joint_ranges] 