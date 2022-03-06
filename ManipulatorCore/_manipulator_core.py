
from ._math import *
from ._joint import Joint

from typing import Literal, List, Tuple
import numpy as np

import warnings
import torch 

from ._exception import ControlException

 
class ManipulatorCore :
    
    def __init__(self,
                 joints : List[Joint], 
                 center_of_masses : np.array = None) :
                 
        correct_type_flag = np.all([isinstance(joint,Joint) for joint in joints])
        
        if not correct_type_flag :
            raise TypeError('Argument joints is expected to be a list of instances of class Joint')
            
        if center_of_masses is not None :
            
            if center_of_masses.shape != (3,len(joints)) :
                raise ValueError('Expected  an array of shape (3,#joints) for the matrix of center of link masses.'
                                 f'Got {center_of_masses.shape}')
            else :
                self.coms = torch.tensor(center_of_masses)
        else :
            self.coms = None
            

        self._joints = joints
                
        #
        #   Initialize arm_matrix and tool_config_vector properties
        #
        
        self._delete_targets()
        
     
    def _delete_targets(self) :
    
        self._arm_matrices = None 
        self_arm_matrix = None
        self._inverse_arm_matrix = None
        self._tool_config_torch =  None
        self._tool_config_vector = None
        self._tool_jacobian = None,
        self._tool_jacobian_second = None
        self._rmrc_matrix = None
        self._linear_link_velocity = None
        self._angular_link_velocity = None
        self._gravity_loading = None
    
    @property
    def arm_matrices(self) :
    
        #return self._arm_matrix[-1].detach().numpy()
        if self._arm_matrices is None :
            self._arm_matrices = self._calculate_arm_matrix()
        return self._arm_matrices
        
    @arm_matrix.setter
    def arm_matrices(self,val) :
        ...
        
    @property
    def arm_matrix(self) :
    
        if self._arm_matrix is None :
            
            if self._arm_matrices is None :
            
                self._arm_matrices = self._calculate_arm_matrix()
            
            self._arm_matrix = self._arm_matrices[-1].detach().numpy()
        
        return self_arm_matrix
           
    @arm_matrix.setter
    def arm_matrix(self) :
        ...
    
    @property
    def inverse_arm_matrix(self) :
        if self._inverse_arm_matrix is None :
            self._inverse_arm_matrix = self._calculate_inverse_arm_matrix()
        return self._inverse_arm_matrix.detach().numpy() #numpy
        
    @inverse_arm_matrix.setter
    def inverse_arm_matrix(self) :
        ...
     
    @property
    def tool_config_torch(self) :
        if self._tool_config_torch is None :
            self._tool_config_torch = self._calculate_tool_config_vector()
        return self._tool_config_torch
           
    @tool_config_vector.setter
    def tool_config_torch(self,val) :
        ...
        
    @property
    def tool_config_vector(self) :
    
        if self._tool_config_vector is None :
        
            if self._tool_config_torch is None :
            
                self._tool_config_torch = self._calculate_tool_config_vector()
                
            self._tool_config_vector = self._tool_config_torch.detach().numpy()
        
        return self._tool_config_vector
        
    @tool_config_vecor.setter
    def tool_config_vector(self) :
        ...
    
    @property
    def tool_jacobian(self) :
        if self._tool_jacobian is None :
            self._tool_jacobian = self._calculate_tool_config_jacobian
        return self._tool_jacobian
        
    @tool_jacobian.setter
    def tool_jacobian(self) :
        ...
        
    @property
    def tool_jacobian_second(self) :
        ...
        
    @tool_jacobian_second.setter
    def tool_jacobian_second(self) :
        ...
        
    @property
    def rmrc_matrix(self) :
        ...
        
    @rmrc_matrix.setter
    def rmrc_matrix(self) :
        if self.rmrc_matrix is None :
            self.rmrc_matrix = self._calculate_rmrc_matrix()
        return self._calculate_rmrc_matrix
    
    @property
    def linear_link_velocity(self) :
        
        if self.coms is None :
            raise AttributeError('Cannot calculate linear link velocities. Centers of masses of the links have not been specified')
        
        if self._linear_link_velocity is None :
            self._linear_link_velocity = self._calculate_linear_link_velocities()
        return self._linear_link_velocity    
        
    @linear_link_velocity.setter(self) :
        ...
        
    @property
    def angular_link_velocity(self) :
    
        if self.coms is None :
            raise AttributeError('Cannot calculate linear link velocities. Centers of masses of the links have not been specified')
            
        if self._angular_link_velocity is None :
            self._angular_link_velocity = self._calculate_angular_link_velocity()
        return self._angular_link_velocity
        
    @angular_link_velocity.setter
    def angular_link_velocity(self) :
        ...
        
    @property
    def gravity_loading(self) :
        ...
        
    @gravity_loading.setter
    def gravity_loading(self) :
        ...
    
    #todo: update to the new concept
    def update_config(self, actuation : List = None, joint_speeds : List = None) :
        
        self._delete_targets()
        
        if acutation is None :  #think about that: a warning is more reasonable 
        
            acuation = np.zeros((len(self.joints),))
            
        elif len(acutation) != len(self._actuation) :
            raise ValueError(f'Setting actuation requires {len(self._joints)} parameters.')
            
        #
        #   TODO: check joint_speeds
        #
        
        for i,joint in enumerate(self._joints) :
            
            joint.set_variable(actuation[i])
            joint.set_speed(joint_speeds[i])
        
    def _calculate_arm_matrix(self) :
        
        local_transforms = [joint.get_transformation() for joint in self._joints]
        
        return matmul(*local_transforms)
        
    def _calculate_inverse_arm_matrix(self) :
    
        with torch.no_grad() :
            rotation = self.arm_matrices[:3,:3]
            translation = self.arm_matrices[:3,3:4]
        
            rot_t = rotation.T 
        
            inverted_transpositon = -1 * rot_t @ translation
        
            tmp = torch.cat([rot_t,inverted_transpositon],axis = 1)
        
            return torch.cat([tmp,torch.tensor([[0,0,0,1]])],axis=0).detach().numpy()
    
    def _calculate_tool_config_vector(self) :
        
        """
            Calculate tool_config_vecor
        """
        
        tool_tip_position = self.arm_matrices[-1][:-1,-1]
        approach_vector = self.arm_matrices[-1][:3,2]

        final_joint = self._joints[-1]
        
        if final_joint.joint_type == 'revolute' :
            approach_vector = torch.exp(final_joint.get_joint_variable() / np.pi) * approach_vector
        
        return torch.cat([tool_tip_position,approach_vector],axis=0)
    
    def _calculate_tool_config_jacobian(self) :
               
        """
            TODO: calculate second order derivative: rather computation intensive: 3D tensor
        """
        grads = []
        
        variables = self._get_joint_variables()
        
        for i in range(6) :
            
            grads.append(torch.autograd.grad(self.tool_config_torch[i],
                                             variables,
                                             retain_graph = (i != 5), 
                                             create_graph = False
                                            )
                        )
           #Remark: zeroing the gradients is already internally handled by autograd
        
        return np.around(np.array(grads),4)
        
    def _calculate_rmrc_matrix(self) :
    
        try : 
    
            return np.linalg.pinv(self.tool_jacobian)
    
        except Exception as e :
        
            raise ControlException(f'Failed to calculate the Moore-Penrose pseudo inverse. Numpy threw: {e}')
    
    def _get_joint_variables(self) :
        
        variables = [joint.get_variable() for joint in self._joints]
        
        return variables
     
    def _get_joint_speeds(self) :
    
        speeds = torch.tensor([joint.get_speed() for joint in self._joints])
        
        return speeds
        
    def  _calculate_linear_link_velocities(self, joint_speeds) :
        
        """
            Calculate the partial derivatives of displacement of the center of mass of each link with respect
            to all joints preceding the link in the kinematic chain
           
        """
        
        variables = self._get_joint_variables()
        
        linear_velocities = torch.zeros((3, len(self.joints)))
        grads = torch.zeros((3,len(self.joints)))
        
        for i in range(len(self.joints)) :
        
            base_com = self.arm_matrices[i] @ self.coms[:,i]
            
            for j in range(3) :
            
                grads[j,:i + 1] = torch.autograd.grad(base_com[j],
                                                      variables[:i + 1],
                                                      retain_graph = (i != 2), 
                                                     )
            linear_velocities[:,i] =  torch.sum(grad * joint_speeds,axis=1)  #broadcasting; an unsqueeze might be required
            
        return linear_velocities.detach().numpy()
        
    def _calculate_angular_link_velocity(self, joint_speeds)
        
        #build mask, where 1 indicates the joint being a revolute joint
        with torch.no_grad() :
            revolute_mask = torch.tensor([joint.is_revolute() for joint in self.joints])
        
            base_rotation_axes = torch.cat([partial_transformation[:3,2 : 3] for partial_transformation in self.arm_matrices])
        
            angular_velocities = torch.zeros((3,len(self.joints)))
        
            for i in range(len(self.joints)) :
            
                #an unsqueezing might be required
                angluar_velocities[:,i] = torch.sum(base_rotation_axes[:,: i + 1] * joint_speeds[:i + 1] * revolute_mask[: i + 1],axis=1)
        
            return angular_velocity.numpy()