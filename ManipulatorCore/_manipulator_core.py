
from typing import Literal, List

from ._math import *
from ._joint import Joint
from ._exception import ControlException
from functools import cached_property
from scipy.linalg import svd

import torch
import numpy as np


class ManipulatorCore:

    """
        TODO: Description

        attributes
        ----------
        TODO

    """
    
    def __init__(self,
                 joints: List[Joint],
                 center_of_masses: np.array = None):
                 
        correct_type_flag = np.all([isinstance(joint, Joint) for joint in joints])
        if not correct_type_flag:
            raise TypeError('Argument joints is expected to be a list of instances of class Joint')
            
        if center_of_masses is not None:
            if center_of_masses.shape != (3, len(joints)):
                raise ValueError('Expected  an array of shape (3,#joints) for the matrix of center of link masses.'
                                  f'Got {center_of_masses.shape}')
            else:
                self.coms = torch.tensor(center_of_masses)
        else:
            self.coms = None

        self.joints = joints

    def _delete_targets(self):
        delattr(self, 'arm_matrices')
        delattr(self, 'arm_matrix')
        delattr(self, 'inverse_arm_matrix')
        delattr(self, 'tool_config_torch')
        delattr(self, 'tool_jacobian')
        delattr(self, 'tool_jacobian_second')
        delattr(self, 'rmrc_matrix')
        delattr(self, 'linear_link_velocity')
        delattr(self, 'angular_link_velocity')
        delattr(self, 'gravity_loading')
    
    @cached_property
    def arm_matrices(self):
        return self._calculate_arm_matrix()
        
    @cached_property
    def arm_matrix(self):
        return self.arm_matrices[-1].detach().numpy()

    @cached_property
    def inverse_arm_matrix(self):
        return self._calculate_inverse_arm_matrix()
     
    @cached_property
    def tool_config_torch(self):
        return self._calculate_tool_config_vector()
        
    @cached_property
    def tool_config_vector(self):
        return self.tool_config_torch.detach().numpy()

    @cached_property
    def tool_jacobian(self):
        return self._calculate_tool_config_jacobian()
        
    @cached_property
    def tool_jacobian_second(self):
        # TODO
        return None

    @cached_property
    def rmrc_matrix(self):
        return self._calculate_rmrc_matrix()

    @cached_property
    def degrees_of_freedom(self):
        return self._calculate_degrees_of_freedom()

    @cached_property
    def linear_link_velocity(self):
        if self.coms is None :
            raise AttributeError('Cannot calculate linear link velocities. Centers of masses of the links have not been specified')
        return self._calculate_linear_link_velocities()
        
    @cached_property
    def angular_link_velocity(self):
        if self.coms is None :
            raise AttributeError('Cannot calculate linear link velocities. Centers of masses of the links have not been specified')
        return self._calculate_angular_link_velocity()

    @cached_property
    def gravity_loading(self):
        # TODO
        return None

    def update_config(self, actuation: List = None, joint_speeds: List = None):
        self._delete_targets()
        if actuation is None :  #think about that: a warning is more reasonable
            actuation = np.zeros((len(self.joints),))
        elif len(actuation) != len(self.actuation) :
            raise ValueError(f'Setting actuation requires {len(self.joints)} parameters.')
            
        #
        #   TODO: check joint_speeds
        #
        for i, joint in enumerate(self.joints):
            joint.set_variable(actuation[i])
            joint.set_speed(joint_speeds[i])
        
    def _calculate_arm_matrix(self):
        local_transforms = [joint.get_transformation() for joint in self.joints]
        return matmul(*local_transforms)
        
    def _calculate_inverse_arm_matrix(self):
    
        with torch.no_grad():
            rotation = self.arm_matrices[-1][:3, :3]
            translation = self.arm_matrices[-1][:3, 3:4]
        
            rot_t = rotation.T
            inverted_transpositon = -1 * rot_t @ translation
            tmp = torch.cat([rot_t, inverted_transpositon], axis=1)
        
            return torch.cat([tmp, torch.tensor([[0, 0, 0, 1]])], axis=0).detach().numpy()
    
    def _calculate_tool_config_vector(self):
        
        """
            Calculate tool_config_vecor
        """
        tool_tip_position = self.arm_matrices[-1][:-1, -1]
        approach_vector = self.arm_matrices[-1][:3, 2]

        final_joint = self.joints[-1]
        if final_joint.joint_type == 'revolute':
            approach_vector = torch.exp(final_joint.get_joint_variable() / np.pi) * approach_vector
        
        return torch.cat([tool_tip_position, approach_vector], axis=0)
    
    def _calculate_tool_config_jacobian(self):
               
        """
            TODO: calculate second order derivative: rather computation intensive: 3D tensor
        """
        grads = []
        variables = self._get_joint_variables()
        
        for i in range(6):
            
            grads.append(torch.autograd.grad(self.tool_config_torch[i],
                                             variables,
                                             retain_graph=(i != 5)))
           #Remark: zeroing the gradients is already internally handled by autograd
        return np.around(np.array(grads), 4)
        
    def _calculate_rmrc_matrix(self):
        try:
            return np.linalg.pinv(self.tool_jacobian)
        except Exception as e:
            raise ControlException(f'Failed to calculate the Moore-Penrose pseudo inverse. Numpy threw: {e}')

    def _calculate_degrees_of_freedom(self):
        u, s, vh = svd(self.tool_jacobian)
        dofs = s.shape[0]
        dof_loss = dofs - min(self.tool_jacobian.shape)
        tipping_point = s[-1]
        return dofs, dof_loss, tipping_point

    def _calculate_linear_link_velocities(self, joint_speeds):

        """
            Calculate the partial derivatives of displacement of the center of mass of each link with respect
            to all joints preceding the link in the kinematic chain

        """

        variables = self.get_joint_variables()

        linear_velocities = torch.zeros((3, len(self.joints)))
        grads = torch.zeros((3, len(self.joints)))

        for i in range(len(self.joints)):

            base_com = self.arm_matrices[i] @ self.coms[:, i]

            for j in range(3):
                grads[j, :i + 1] = torch.autograd.grad(base_com[j],
                                                       variables[:i + 1],
                                                       retain_graph=(i != 2))
            linear_velocities[:, i] = torch.sum(grads * joint_speeds,
                                                axis=1)  # broadcasting; an unsqueeze might be required

        return linear_velocities.detach().numpy()

    def _calculate_angular_link_velocity(self, joint_speeds):

        # build mask, where 1 indicates the joint being a revolute joint
        with torch.no_grad():
            revolute_mask = torch.tensor([joint.is_revolute() for joint in self.joints])

            base_rotation_axes = torch.cat(
                [partial_transformation[:3, 2: 3] for partial_transformation in self.arm_matrices])

            angular_velocities = torch.zeros((3, len(self.joints)))

            for i in range(len(self.joints)):
                # an unsqueezing might be required
                angular_velocities[:, i] = torch.sum(
                    base_rotation_axes[:, : i + 1] * joint_speeds[:i + 1] * revolute_mask[: i + 1], axis=1)

            return angular_velocities.numpy()

    def _get_joint_variables(self):
        variables = [joint.get_variable() for joint in self.joints]
        return variables
     
    def _get_joint_speeds(self):
        speeds = torch.tensor([joint.get_speed() for joint in self.joints])
        return speeds

