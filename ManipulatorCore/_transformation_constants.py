
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
    # x-rotation
    tns([
            [0, 0, 0, 0], 
            [0, 0, -1, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    # y-rotation
    tns([
            [0, 0, 1, 0], 
            [0, 0, 0, 0], 
            [-1, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    # z-rotation
    tns([
            [0, -1, 0, 0], 
            [1, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float)  
]

cos_mask = [
    # x-rotation
    tns([
            [0, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    # y-rotation
    tns([
            [1, 0, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 1, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float),
    # z-rotation
    tns([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 0, 0], 
            [0, 0, 0, 0] 
        ], dtype=torch.float)  
]

trans_mask = [
    # x-translation
    tns([
        [0, 0, 0, 1], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0] 
    ], dtype=torch.float), 
    # y-translation
    tns([
        [0, 0, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0], 
        [0, 0, 0, 0] 
    ], dtype=torch.float), 
    # z-translation
    tns([
        [0, 0, 0, 0], 
        [0, 0, 0, 0], 
        [0, 0, 0, 1], 
        [0, 0, 0, 0] 
    ], dtype=torch.float)  
]
