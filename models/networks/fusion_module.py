import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone

import pdb


class PointCatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel):
        super(PointCatFusion, self).__init__()
        self.in_channel_list = in_channel_list
        self.out_channel = out_channel

        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        c_mid = max(s // 2, out_channel)
        self.merge_layer = nn.Sequential(
            backbone.conv1x1_bn_relu(s, c_mid),
            backbone.conv1x1_bn_relu(c_mid, out_channel)
        )
    
    def forward(self, *x_list):
        #pdb.set_trace()
        x_merge = torch.cat(x_list, dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out


class CatFusion(nn.Module):
    def __init__(self, in_channel_list, out_channel, double_branch=True):
        super(CatFusion, self).__init__()
        self.double_branch = double_branch
        if self.double_branch:
            out_channel = 2 * out_channel
        
        self.in_channel_list = in_channel_list
        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        self.fusion_net = nn.Sequential(
            backbone.conv3x3_bn_relu(s, out_channel),
            backbone.conv3x3_bn_relu(out_channel, out_channel)
        )
    
    def forward(self, *x_list):
        x_cat = torch.cat(x_list, dim=1)
        x_out = self.fusion_net(x_cat)
        if self.double_branch:
            x1, x2 = x_out.chunk(2, dim=1)
            return x1.contiguous(), x2.contiguous()
        else:
            return x_out.contiguous(), x_out.contiguous()


class CatFusionCtx(nn.Module):
    def __init__(self, in_channel_list, out_channel, double_branch=True):
        super(CatFusionCtx, self).__init__()
        self.double_branch = double_branch
        if self.double_branch:
            out_channel = 2 * out_channel
        
        self.in_channel_list = in_channel_list
        assert len(self.in_channel_list) >= 2

        s = 0
        for in_channel in self.in_channel_list:
            s = s + in_channel
        
        cmid = s // 3
        self.conv1 = backbone.conv3x3_bn_relu(s, cmid, dilation=1)
        self.conv2 = backbone.conv3x3_bn_relu(cmid, cmid, dilation=2)
        self.conv4 = backbone.conv3x3_bn_relu(cmid, cmid, dilation=4)

        self.conv_merge = backbone.conv3x3_bn_relu(3 * cmid, out_channel, dilation=1)
    
    def forward(self, *x_list):
        x_cat = torch.cat(x_list, dim=1)

        x_cat_1 = self.conv1(x_cat)
        x_cat_2 = self.conv2(x_cat_1)
        x_cat_4 = self.conv4(x_cat_2)

        x_merge = torch.cat((x_cat_1, x_cat_2, x_cat_4), dim=1)
        x_out = self.conv_merge(x_merge)
        if self.double_branch:
            x1, x2 = x_out.chunk(2, dim=1)
            return x1.contiguous(), x2.contiguous()
        else:
            return x_out.contiguous(), x_out.contiguous()