import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lib

import pdb


class VMPModule(nn.Module):
    def __init__(self, output_size, scale_rate=None):
        super(VMPModule, self).__init__()
        self.output_size = output_size
        self.scale_rate = scale_rate
        if self.scale_rate is None:
            self.scale_rate = [1 for i in range(len(self.output_size))]
    
    def forward(self, pcds_feat, pcds_ind):
        voxel_feat = pytorch_lib.VoxelMaxPool(pcds_feat=pcds_feat, pcds_ind=pcds_ind, output_size=self.output_size, scale_rate=self.scale_rate)
        return voxel_feat