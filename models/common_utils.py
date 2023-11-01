import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks import backbone
import deep_point


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    voxel_feat = deep_point.VoxelMaxPool(pcds_feat=pcds_feat.float(), pcds_ind=pcds_ind, output_size=output_size, scale_rate=scale_rate).to(pcds_feat.dtype)
    return voxel_feat


def reproj(pcds_xyzi, Voxel, dx, dy, phi_range_radian, theta_range_radian, dphi, dtheta):
    '''
    Input:
        pcds_xyzi (BS, N, 3), 3 -> (x, y, z)
    Output:
        pcds_coord_wl_reproj, pcds_sphere_coord_reproj (BS, N, 2, 1)
    '''
    # bev quat (BS, N)
    x = pcds_xyzi[:, :, 0]
    y = pcds_xyzi[:, :, 1]
    z = pcds_xyzi[:, :, 2]
    d = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt() + 1e-12

    x_quan = (x - Voxel.range_x[0]) / dx
    y_quan = (y - Voxel.range_y[0]) / dy
    pcds_coord_wl_reproj = torch.stack((x_quan, y_quan), dim=-1).unsqueeze(-1)

    # rv quat
    phi = phi_range_radian[1] - torch.atan2(x, y)
    phi_quan = (phi / dphi)
    
    theta = theta_range_radian[1] - torch.asin(z / d)
    theta_quan = (theta / dtheta)
    pcds_sphere_coord_reproj = torch.stack((theta_quan, phi_quan), dim=-1).unsqueeze(-1)
    return pcds_coord_wl_reproj, pcds_sphere_coord_reproj


def reproj_with_offset(pcds_xyzi, pcds_offset, Voxel, dx, dy, phi_range_radian, theta_range_radian, dphi, dtheta):
    '''
    Input:
        pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
        pcds_offset (BS, N, 3)
    Output:
        pcds_coord_wl_reproj, pcds_sphere_coord_reproj (BS, N, 2, 1)
    '''
    # bev quat (BS, N, 1)
    x = pcds_xyzi[:, 0, :] + pcds_offset[:, :, [0]]
    y = pcds_xyzi[:, 1, :] + pcds_offset[:, :, [1]]
    z = pcds_xyzi[:, 2, :] + pcds_offset[:, :, [2]]
    d = (x.pow(2) + y.pow(2) + z.pow(2)).sqrt() + 1e-12

    x_quan = (x - Voxel.range_x[0]) / dx
    y_quan = (y - Voxel.range_y[0]) / dy
    pcds_coord_wl_reproj = torch.stack((x_quan, y_quan), dim=2)

    # rv quat
    phi = phi_range_radian[1] - torch.atan2(x, y)
    phi_quan = (phi / dphi)
    
    theta = theta_range_radian[1] - torch.asin(z / d)
    theta_quan = (theta / dtheta)
    pcds_sphere_coord_reproj = torch.stack((theta_quan, phi_quan), dim=2)
    return pcds_coord_wl_reproj, pcds_sphere_coord_reproj


class CatFusion(nn.Module):
    def __init__(self, cin1, cin2, cout, double_branch=True):
        super(CatFusion, self).__init__()
        out_channels = cout
        self.double_branch = double_branch
        if self.double_branch:
            out_channels = 2 * cout
        
        self.fusion_net = nn.Sequential(
            backbone.conv3x3(cin1 + cin2, out_channels, stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            backbone.act_layer,

            backbone.conv3x3(out_channels, out_channels, stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            backbone.act_layer
        )
    
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)
        x_out = self.fusion_net(x_cat)
        if self.double_branch:
            return x_out.chunk(2, dim=1)
        else:
            return x_out, x_out


class CatFusionASPP(nn.Module):
    def __init__(self, cin1, cin2, cout, double_branch=True):
        super(CatFusionASPP, self).__init__()
        out_channels = cout
        self.double_branch = double_branch
        if self.double_branch:
            out_channels = 2 * cout
        
        cmid = (cin1 + cin2) // 3
        self.conv1 = backbone.conv3x3(cin1 + cin2, cmid, stride=1, dilation=1)
        self.conv3 = backbone.conv3x3(cin1 + cin2, cmid, stride=1, dilation=3)
        self.conv6 = backbone.conv3x3(cin1 + cin2, cmid, stride=1, dilation=6)

        self.conv_merge = nn.Sequential(
            nn.BatchNorm2d(3 * cmid),
            backbone.act_layer,

            backbone.conv3x3(3 * cmid, out_channels, stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            backbone.act_layer
        )
    
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)

        x_merge = torch.cat((self.conv1(x_cat), self.conv3(x_cat), self.conv6(x_cat)), dim=1)
        x_out = self.conv_merge(x_merge)
        if self.double_branch:
            return x_out.chunk(2, dim=1)
        else:
            return x_out, x_out


class CatFusionCtx(nn.Module):
    def __init__(self, cin1, cin2, cout, double_branch=True):
        super(CatFusionCtx, self).__init__()
        out_channels = cout
        self.double_branch = double_branch
        if self.double_branch:
            out_channels = 2 * cout
        
        cmid = (cin1 + cin2) // 3
        self.conv1 = nn.Sequential(
            backbone.conv3x3(cin1 + cin2, cmid, stride=1, dilation=1),
            nn.BatchNorm2d(cmid),
            backbone.act_layer
        )
        self.conv2 = nn.Sequential(
            backbone.conv3x3(cmid, cmid, stride=1, dilation=2),
            nn.BatchNorm2d(cmid),
            backbone.act_layer
        )
        self.conv4 = nn.Sequential(
            backbone.conv3x3(cmid, cmid, stride=1, dilation=4),
            nn.BatchNorm2d(cmid),
            backbone.act_layer
        )

        self.conv_merge = nn.Sequential(
            backbone.conv3x3(3 * cmid, out_channels, stride=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            backbone.act_layer
        )
    
    def forward(self, x1, x2):
        x_cat = torch.cat((x1, x2), dim=1)

        x_cat_1 = self.conv1(x_cat)
        x_cat_2 = self.conv2(x_cat_1)
        x_cat_4 = self.conv4(x_cat_2)

        x_merge = torch.cat((x_cat_1, x_cat_2, x_cat_4), dim=1)
        x_out = self.conv_merge(x_merge)
        if self.double_branch:
            return x_out.chunk(2, dim=1)
        else:
            return x_out, x_out