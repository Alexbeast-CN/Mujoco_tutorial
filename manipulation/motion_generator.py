from ruckig import InputParameter, OutputParameter, Result, Ruckig
import numpy as np
from .joint_utils import JointInfo

class MotionGenerator:
    def __init__(self, joint_info: JointInfo, timestep: float):
        """初始化轨迹规划器
        
        Args:
            joint_info: 关节信息对象
            timestep: 时间步长
        """
        self.joint_info = joint_info
        self.dof = joint_info.dof
        self.inp = InputParameter(self.dof)
        self.out = OutputParameter(self.dof)
        self.otg = Ruckig(self.dof, timestep)
        
        # 使用关节信息中的限制配置
        if joint_info.velocity_limits:
            self.inp.max_velocity = joint_info.velocity_limits
        else:
            # 使用默认值
            self.inp.max_velocity = [1.5] * self.dof
            
        if joint_info.acceleration_limits:
            self.inp.max_acceleration = joint_info.acceleration_limits
        else:
            self.inp.max_acceleration = [v * 2 for v in self.inp.max_velocity]
            
        if joint_info.jerk_limits:
            self.inp.max_jerk = joint_info.jerk_limits
        else:
            self.inp.max_jerk = [a * 2 for a in self.inp.max_acceleration]
        
        # 初始化状态
        self.inp.current_position = [0.0] * self.dof
        self.inp.current_velocity = [0.0] * self.dof
        self.inp.current_acceleration = [0.0] * self.dof
        self.inp.target_position = [0.0] * self.dof
        self.inp.target_velocity = [0.0] * self.dof
        self.inp.target_acceleration = [0.0] * self.dof
        
        self.reach_target = True
        self.ruckig_result = Result.Finished
        
    def update_current_position(self, positions):
        """更新当前位置
        
        Args:
            positions: 当前关节位置列表
        """
        self.inp.current_position = positions.copy()
        
    def set_joint_position(self, positions):
        """设置目标位置
        
        Args:
            positions: 目标关节位置列表
            
        Returns:
            bool: 如果目标位置有效返回True，否则返回False
        """
        # 检查位置限制
        for i, pos in enumerate(positions):
            if not (self.joint_info.joint_ranges[i][0] <= pos <= self.joint_info.joint_ranges[i][1]):
                print(f"警告：关节 {self.joint_info.joint_names[i]} 的目标位置超出限制范围")
                return False
                
        self.inp.target_position = positions.copy()
        self.reach_target = False
        self.ruckig_result = Result.Working
        return True
        
    def update(self):
        """更新轨迹规划
        
        Returns:
            new_positions: 新的关节位置
            reach_target: 是否到达目标
        """
        if self.ruckig_result == Result.Working:
            self.ruckig_result = self.otg.update(self.inp, self.out)
            self.out.pass_to_input(self.inp)
            
        if self.ruckig_result == Result.Finished:
            self.reach_target = True
            
        return self.out.new_position, self.reach_target 