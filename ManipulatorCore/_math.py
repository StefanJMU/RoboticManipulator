
from typing import Union
from torch.autograd import Variable
from torch import tensor
from ._transformation_constants import *


def _rot(angle, axis: int):
    
    """
        fundamental homogenous rotation matrix
    """
    cos, sin = torch.cos(angle), torch.sin(angle)
    return sockets[axis] + cos_mask[axis]*cos + sin_mask[axis]*sin


def _tran(translation, axis: int):
    
    """
        fundamental homogenous translation matrix
    """
    return sockets[-1] + translation*trans_mask[axis]


def transformation_from_kinematic_parameters(theta: Union[tensor, Variable],
                                             d: Union[tensor, Variable],
                                             a: Union[tensor, Variable],
                                             alpha: Union[tensor, Variable]):
    
    """
        Transformation matrix from kinematic parameters
        
        parameters
        ----------
        theta : float, torch.tensor, torch.autograd.Variable
            Denavit-Hartenberg parameter : joint angle
        d : float, torch.tensor, torch.autograd.Variable
            Denavit-Hartenberg parameter : joint length
        a : float, torch.tensor, torch.autograd.Variable
            Denavit-Hartenberg parameter : link length
        alpha : float, torch.tensor, torch.autograd.Variable
            Denavit-Hartenberg parameter : link twist angle
    """
    matrices = [
            
            _rot(theta, 2),
            _tran(d, 2),
            _tran(a, 0),
            _rot(alpha, 0),
    ]
    return matmul(*matrices, retain_intermediates=False)


def screw_rotation(axis: int,
                   angle: Union[tensor, Variable],
                   translation: Union[tensor, Variable]):

    """
        Calculates Screw rotation (rotation and translation along the same axis)

        parameters
        ----------
        axis : int
            axis along the screw rotation is conducted
        angle: float, torch.tensor, torch.autograd.Variable
            angle by which the system rotates
        translation: float, torch.tensor, torch.autograd.Variable
            translation by which the system translates
    """
    rotation = _rot(angle, axis)
    translation = _tran(translation, axis)

    return rotation @ translation


def matmul(*matrices, retain_intermediates=True):
    
    """
        Utility function : multiply matrices from left to right
        Retains intermediate results if specififed

        parameters
        ----------

        matrices : list
            List of np.array matrices to be multiplied
        retain_intermediates : bool
            Flag indicating the retaining of intermediate matrix multiplication results

        returns
        -------
        resulting np.array matrix or list of np.array matrices
    """
    intermediates = [matrices[0]]
    for i in range(1, len(matrices)):
        intermediates.append(intermediates[-1] @ matrices[i])

    return intermediates if retain_intermediates else intermediates[-1]

