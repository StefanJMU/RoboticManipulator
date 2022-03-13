from typing import Literal, List, Tuple
import numpy as np

import torch
from torch import tensor as tns
from torch.autograd import Variable

from ._math import *


class Joint:

    """
        TODO: docstring

    """
    
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
        self.is_revolute = joint_type == 'revolute'

        if joint_type == 'revolute':
            self.joint_angle = Variable(tns(self._sanitize_angle(joint_angle), dtype=torch.float),
                                        requires_grad=True)
            self.joint_length = self._sanitize_length(joint_length, 'joint_length')
            self.generalized_var = self.joint_angle  # ref
        else:
            self.joint_angle = tns(self._sanitize_angle(joint_angle))
            self.joint_length = Variable(tns(self._sanitize_length(joint_length, 'joint_length'), dtype=torch.float),
                                         requires_grad=True)
            self.generalized_var = self.joint_length  # ref

        self.link_twist_angle = tns(self._sanitize_angle(link_twist_angle))
        self.link_length = self._sanitize_length(link_length, 'link_length')
        self._link_screw = screw_rotation(0,
                                          tns(self._sanitize_angle(link_twist_angle)),
                                          self._sanitize_length(link_length, 'link_length'))

    def _sanitize_length(self, length, arg_name):
        if length < 0:
            raise ValueError(f'Expected a non-negative value of argument {arg_name}.'
                             f'Got {length}')
        return length
        
    def _sanitize_angle(self, angle):
        if self.angle_unit == 'degrees':
            return (angle * np.pi) / 180
        return angle

    def set_variable(self, value: float):
        with torch.no_grad():
            if self.is_revolute:
                self.joint_angle.fill_(self._sanitize_angle(value))
            else:
                self.joint_length.fill_(self._sanitize_length(value, 'joint_length'))

    def get_transformation(self):
        joint_screw = screw_rotation(2,
                              self.joint_angle,
                              self.joint_length)
        return joint_screw @ self._link_screw
        
    def get_variable(self):
        return self.generalized_var

    def get_joint_position(self):
        return self.generalized_var.item()
    
    