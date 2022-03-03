
from ._math import *
from ._joint import Joint

from typing import Literal, List, Tuple
import numpy as np

import warnings
import torch 

from ._exception import ControlException

 
class ManipulatorCore :
    
    def __init__(self, joints : List[Joint]) :
        
        correct_type_flag = np.all([isinstance(joint,Joint) for joint in joints])
        
        if not correct_type_flag :
            raise TypeError('Argument joints is expected to be a list of instances of class Joint')
            

        self._joints = joints
                
        #
        #   Initialize arm_matrix and tool_config_vector properties
        #
        self._update_targets()
        
        self.arm_matrix = self.arm_matrix
        self.tool_config_vector = self.tool_config_vector
       
       
    @property
    def arm_matrix(self) :
    
        return self._arm_matrix.detach().numpy()
        
    @property
    def tool_config_vector(self) :
        return self._tool_config_vector.detach().numpy()
        
    @arm_matrix.setter
    def arm_matrix(self,val) :
        ...
        
    @tool_config_vector.setter
    def tool_config_vector(self,val) :
        ...
        
    def _update_targets(self) :
        
        """
            TODO : make this configurable.
            The calling order must not be changed
        """
        self._arm_matrix = self._calculate_arm_matrix() 
        self.inverse_arm_matrix = self._calculate_inverse_arm_matrix()#numpy
        self._tool_config_vector =  self._calculate_tool_config_vector()
        self.tool_jacobian,self.tool_jacobian_second = self._calculate_tool_config_jacobian() #numpy
        self.rmrc_matrix = self._calculate_rmrc_matrix() #numpy
    
    def update_config(self, actuation : List = None) :
        
        if acutation is None :  #think about that: a warning is more reasonable 
        
            acuation = np.zeros((len(self.joints),))
            
        elif len(acutation) != len(self._actuation) :
            raise ValueError(f'Setting actuation requires {len(self._joints)} parameters.')
        
        for i,joint in enumerate(self._joints) :
            
            joint.set_variable(actuation[i])
            
        self._update_targets()
        
    def _calculate_arm_matrix(self) :
        
        """
            Retrieve the arm matrix as possibly cached property (tbd)
        """
        local_transforms = [joint.get_transformation() for joint in self._joints]
        
        return matmul(*local_transforms)
        
    def _calculate_inverse_arm_matrix(self) :
    
        rotation = self._arm_matrix[:3,:3]
        translation = self._arm_matrix[:3,3:4]
        
        rot_t = rotation.T 
        
        inverted_transpositon = -1 * rot_t @ translation
        
        tmp = torch.cat([rot_t,inverted_transpositon],axis = 1)
        
        return torch.cat([tmp,torch.tensor([[0,0,0,1]])],axis=0).detach().numpy()
    
    def _calculate_tool_config_vector(self) :
        
        """
            Calculate tool_config_vecor
        """
        
        tool_tip_position = self._arm_matrix[:-1,-1]
        approach_vector = torch.matmul(self._arm_matrix[:3,:3],tns([0,0,1],dtype=torch.float).T)

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
            
            grads.append(torch.autograd.grad(self._tool_config_vector[i],
                                             variables,
                                             retain_graph = (i != 5), 
                                             create_graph = False
                                            )
                        )
           #Remark: zeroing the gradients is already internally handled by autograd
        
        return np.around(np.array(grads),4), None
        
    def _calculate_rmrc_matrix(self) :
    
        try : 
    
            return np.linalg.pinv(self.tool_jacobian)
    
        except Exception as e :
        
            raise ControlException(f'Failed to calculate the Moore-Penrose pseudo inverse. Numpy threw: {e}')
    
    def _get_joint_variables(self) :
        
        variables = [joint.get_variable() for joint in self._joints]
        
        return variables
            
        
