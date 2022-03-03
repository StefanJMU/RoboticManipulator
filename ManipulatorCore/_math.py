
import numpy as np
import torch
from torch import tensor as tns

from ._transformation_constants import *

def rot(angle, axis : int) :
    
    """
        fundamental homogenous rotation matrix
    """
    if axis not in [0,1,2] :
        raise ValueError('Fundamental rotation matrix axis are selected from {0,1,2}.'
                         f'Got {axis}')
                         
    cos,sin = torch.cos(angle), torch.sin(angle)
    return sockets[axis] + cos_mask[axis] * cos + sin_mask[axis] * sin

def tran(x_tran = ZERO_1D, y_tran = ZERO_1D, z_tran = ZERO_1D) :
    
    """
        fundamental homogenous translation matrix
    """
    return sockets[-1] + x_tran * trans_mask[0] + y_tran * trans_mask[1] + z_tran * trans_mask[2]

def transformation_from_kinematic_parameters(theta,d,a,alpha) :
    
    """
        Transformation matrix from kinematic parameters
        
        parameters
        ----------
        theta : float
            Denavit-Hartenberg parameter : joint angle
        d : float
            Denavit-Hartenberg parameter : joint length
        a : float
            Denavit-Hartenberg parameter : link length
        alpha : float
            Denavit-Hartenberg parameter : link twist angle
    """
    matrices = [
            
            rot(theta,2),
            tran(ZERO_1D,ZERO_1D,d),
            tran(a),
            rot(alpha,0),
    ]
    
    return matmul(*matrices)

def matmul(*matrices) :
    
    """
        Utility function : multiply matrices from left to right
    """
    matrix = matrices[0]
    for i in range(1,len(matrices)) :
        matrix = torch.matmul(matrix,matrices[i])
    
    return matrix