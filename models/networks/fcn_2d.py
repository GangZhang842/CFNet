import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
from utils.config_parser import get_module

import pdb


class FCN2D(nn.Module):
    def __init__(self, base_block, fpn_block, base_channels, base_layers, base_strides, base_dilation=None, double_branch=True):
        super(FCN2D, self).__init__()
        self.base_block = base_block
        self.fpn_block = fpn_block
        self.base_channels = base_channels
        self.base_layers = base_layers
        self.base_strides = base_strides
        self.base_dilation = base_dilation
        if self.base_dilation is None:
            self.base_dilation = [1 for i in range(len(self.base_strides))]
        
        assert (len(self.base_channels) - 1) == len(self.base_layers)
        assert len(self.base_layers) == len(self.base_strides)
        assert len(self.base_strides) == len(self.base_dilation)

        #encoder
        self.enc_net = nn.ModuleList()
        for i in range(len(self.base_layers)):
            self.enc_net.append(self._make_layer(self.base_block, self.base_channels[i], self.base_channels[i + 1],
            self.base_layers[i], stride=self.base_strides[i], dilation=self.base_dilation[i]))
        
        #decoder
        self.dec_net = nn.ModuleList()
        L = len(self.base_channels)
        fusion_channels_in = self.base_channels[L - 1]
        fusion_channels_out = None
        for i in range(1, len(self.base_layers)):
            fusion_channels_out = (fusion_channels_in + self.base_channels[L - 1 - i]) // 2
            self.dec_net.append(get_module(type=self.fpn_block, cin_low=self.base_channels[L - 1 - i],
            cin_high=fusion_channels_in, cout=fusion_channels_out, scale_factor=self.base_strides[L - 1 - i]))

            fusion_channels_in = fusion_channels_out
        
        self.out_channels = fusion_channels_out

        self.double_branch = double_branch
        if self.double_branch:
            self.sem_ins_conv = backbone.conv3x3_bn_relu(self.out_channels, 2 * self.out_channels, stride=1, dilation=1)
    
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilation=1):
        layer = []
        layer.append(backbone.DownSample2D(in_planes, out_planes, stride=stride))
        
        for i in range(num_blocks):
            layer.append(get_module(type=block, in_planes=out_planes, dilation=dilation, use_att=False))
        
        layer.append(get_module(type=block, in_planes=out_planes, dilation=dilation, use_att=True))
        return nn.Sequential(*layer)
    
    def forward(self, x):
        #pdb.set_trace()
        #encoder
        x_list = [self.enc_net[0](x)]
        for i in range(1, len(self.enc_net)):
            x_list.append(self.enc_net[i](x_list[-1]))
        
        #decoder
        L = len(x_list)
        x_merge = self.dec_net[0](x_list[L - 2], x_list[L - 1])
        for i in range(1, len(self.dec_net)):
            x_merge = self.dec_net[i](x_list[L - 2 - i], x_merge)
        
        if self.double_branch:
            x_sem, x_ins = self.sem_ins_conv(x_merge).chunk(2, dim=1)
            return x_sem.contiguous(), x_ins.contiguous()
        else:
            return x_merge.contiguous(), x_merge.contiguous()