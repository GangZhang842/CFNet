import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone

import pdb


class Merge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(Merge, self).__init__()
        cin = cin_low + cin_high
        self.scale_factor = scale_factor

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.merge_layer = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            backbone.conv3x3_bn_relu(cin, cin // 2, stride=1, dilation=1),
            backbone.conv3x3_bn_relu(cin // 2, cout, stride=1, dilation=1)
        )
    
    def forward(self, x_low, x_high):
        x_high_up = self.upsample(x_high)
        x_merge = torch.cat((x_low, x_high_up), dim=1)
        x_out = self.merge_layer(x_merge)
        return x_out


class AttMerge(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMerge, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            backbone.conv3x3_bn_relu(2 * cout, cout // 2, stride=1, dilation=1),
            backbone.conv3x3(cout // 2, 1, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )

        self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        x_high_up = self.upsample(x_high)

        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high_up)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)

        # attention fusion
        ca_map = self.att_layer(x_merge)
        x_out = x_low_feat * ca_map + x_high_up_feat * (1 - ca_map)
        return x_out


class AttMergeDeConv(nn.Module):
    def __init__(self, cin_low, cin_high, cout, scale_factor):
        super(AttMergeDeConv, self).__init__()
        self.scale_factor = scale_factor
        self.cout = cout

        self.dropout = nn.Dropout(p=0.2, inplace=False)
        self.att_layer = nn.Sequential(
            backbone.conv3x3_bn_relu(2 * cout, cout // 2, stride=1, dilation=1),
            backbone.conv3x3(cout // 2, 1, stride=1, dilation=1, bias=True),
            nn.Sigmoid()
        )

        self.conv_high = None
        if (scale_factor == 1) or (isinstance(scale_factor, (tuple, list)) and all([x == 1 for x in scale_factor])):
            self.conv_high = backbone.conv3x3_bn_relu(cin_high, cout, stride=1, dilation=1)
        else:
            self.conv_high = nn.Sequential(
                backbone.DeConv(cin_high, cout, stride=self.scale_factor, groups=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True)
            )
        self.conv_low = backbone.conv3x3_bn_relu(cin_low, cout, stride=1, dilation=1)
    
    def forward(self, x_low, x_high):
        #pdb.set_trace()
        x_low_feat = self.conv_low(x_low)
        x_high_up_feat = self.conv_high(x_high)

        x_merge = torch.cat((x_low_feat, x_high_up_feat), dim=1) #(BS, 2*channels, H, W)
        x_merge = self.dropout(x_merge)

        # attention fusion
        ca_map = self.att_layer(x_merge)
        x_out = x_low_feat * ca_map + x_high_up_feat * (1 - ca_map)
        return x_out