import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lib

import pdb


class G2PModule(nn.Module):
    def __init__(self, scale_rate):
        super(G2PModule, self).__init__()
        self.scale_rate = scale_rate
    
    def forward(self, grid_in, pcds_ind):
        pcds_feat = pytorch_lib.Grid2Point(grid_in, pcds_ind, scale_rate=self.scale_rate)
        return pcds_feat