from typing import Literal, List, Tuple
import typing
import numpy as np

import torch
from torch import tensor as tns
from torch.autograd import Variable

from ._math import *


class Joint:
    
    def __init__(self,
                 joint_type: Literal['revolute', 'prismatic'],
                 joint_angle: float = 0,
                 joint_length: float = 0,
                 link_length: float = 0,
                 link_twist_angle: float = 0,
                 angle_unit: Literal['degrees', 'radians'] = 'radians'):
        
        if joint_type not in ['revolute', 'prismatic']:
            raise ValueError("joint_type is required to be 'prismatic' or 'revolute'."
                             f'Got {joint_type}')
                             
        if angle_unit not in ['degrees', 'radians']:
            raise ValueError('angle_unit is required to be in [degrees,radians].'
                             f'Got {angle_unit}')
        
        self.joint_speed = 0
        self.angle_unit = angle_unit
        self.joint_type = joint_type
        
        if joint_type == 'revolute':
            self.joint_angle = Variable(tns([self.sanitize_angle(joint_angle)], dtype=torch.float),
                                        requires_grad=True)
            self.joint_length = self.sanitize_length(joint_length, 'joint_length')
            self.generalized_var = self.joint_angle  # ref
        else :
            self.joint_angle = tns(self.sanitize_angle(joint_angle))
            self.joint_length = Variable(tns([self.sanitize_length(joint_length, 'joint_length')], dtype=torch.float),
                                         requires_grad=True)
            self.generalized_var = self.joint_length  # ref
        
        self.link_length = self.sanitize_length(link_length, 'link_length')
        self.link_twist_angle = tns(self.sanitize_angle(link_twist_angle))

    def sanitize_length(self, length, arg_name):
        if length < 0:
            raise ValueError(f'Expected a non-negative value of argument {arg_name}.'
                             f'Got {length}')
        return length
        
    def sanitize_angle(self, angle):
        if self.angle_unit == 'degrees':
            return (angle * np.pi) / 180
        return angle
    
    def set_variable(self, value: float):
        with torch.no_grad():
            if self.joint_type == 'revolute':
                self.joint_angle[0] = self.sanitize_angle(value)
            else:
                self.joint_length[0] = self.sanitize_length(value, 'joint_length')
            
    def set_joint_speed(self, value: float):
        self.joint_speed = value
        
    def get_speed(self):
        return self.joint_speed
        
    def is_revolute(self):
        if self.joint_type == 'revolute':
            return 1
        return 0

    def get_transformation(self):
        return transformation_from_kinematic_parameters(self.joint_angle,
                                                        self.joint_length,
                                                        self.link_length, 
                                                        self.link_twist_angle)
        
    def get_variable(self):
        return self.generalized_var

    def get_joint_position(self):
        return self.generalized_var[0].item()
    
    