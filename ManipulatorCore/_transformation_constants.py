
import torch
from torch import tensor as tns

sockets = [
        # x-rotation socket-matrix
        tns([
            [1,  0,  0,  0], 
            [0,  0,  0,  0], 
            [0,  0,  0,  0], 
            [0,  0,  0,  1]
        ],  dtype=torch.float),
        
        # y-rotation socket-matrix
        tns([
            [0, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 1]
        ], dtype=torch.float),
    
        # z-rotation socket-matrix
        tns([
            [0, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], dtype=torch.float),
    
        # translation socket-matrix
        tns([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 1]
        ], dtype=torch.float)
]

sin_mask = [
    
    tns([
            [0, 0, 0, 0], 
            [0, 0, -1, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    
    tns([
            [0, 0, 1, 0], 
            [0, 0, 0, 0], 
            [-1, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    
    tns([
            [0, -1, 0, 0], 
            [1, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float)  
]

cos_mask = [
    
    tns([
            [0, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float), 
    
    tns([
            [1, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float), 
    
    tns([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float)  
]

trans_mask = [
    
    tns([
        [0, 0, 0, 1], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0] 
    ], dtype=torch.float), 
    
    tns([
        [0, 0, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0] 
    ], dtype=torch.float), 
    
    tns([
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0] 
    ], dtype=torch.float)  
]

ZERO_1D = tns([0], dtype=torch.float)
