import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from .networks import backbone
from . import common_utils

from utils.config_parser import get_module

import pdb


class CFNet(nn.Module):
    def __init__(self, pModel):
        super(CFNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(self.pModel.Voxel.bev_shape)
        self.rv_shape = list(self.pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (self.pModel.Voxel.range_x[1] - self.pModel.Voxel.range_x[0]) / self.pModel.Voxel.bev_shape[0]
        self.dy = (self.pModel.Voxel.range_y[1] - self.pModel.Voxel.range_y[0]) / self.pModel.Voxel.bev_shape[1]

        self.phi_range_radian = (-np.pi, np.pi)
        self.theta_range_radian = (self.pModel.Voxel.RV_theta[0] * np.pi / 180.0, self.pModel.Voxel.RV_theta[1] * np.pi / 180.0)

        self.dphi = (self.phi_range_radian[1] - self.phi_range_radian[0]) / self.pModel.Voxel.rv_shape[1]
        self.dtheta = (self.theta_range_radian[1] - self.theta_range_radian[0]) / self.pModel.Voxel.rv_shape[0]

        self.point_feat_out_channels = self.pModel.point_feat_out_channels
        self.build_network()
    
    def build_network(self):
        # build network
        bev_net_cfg = self.pModel.BEVParam
        rv_net_cfg = self.pModel.RVParam
        bev_base_channels = bev_net_cfg.base_channels

        fusion_cfg = self.pModel.FusionParam

        # base network
        self.point_pre = nn.Sequential(
            backbone.bn_conv1x1_bn_relu(7, bev_base_channels[0]),
            backbone.conv1x1_bn_relu(bev_base_channels[0], bev_base_channels[0])
        )

        # BEV network
        self.point2bev = get_module(bev_net_cfg.P2VParam)
        self.bev_net = get_module(bev_net_cfg)
        self.bev2point = get_module(bev_net_cfg.V2PParam)

        # RV network
        self.point2rv = get_module(rv_net_cfg.P2VParam)
        self.rv_net = get_module(rv_net_cfg)
        self.rv2point = get_module(rv_net_cfg.V2PParam)

        # stage0
        # sem branch
        point_fusion_channels = (bev_base_channels[0], self.bev_net.out_channels, self.rv_net.out_channels)
        self.point_fusion_sem = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
        self.pred_layer_sem = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

        # ins branch
        self.point_fusion_ins = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
        self.pred_layer_offset = backbone.PredBranch(self.point_feat_out_channels, 3)
        self.pred_layer_hmap = nn.Sequential(
            backbone.PredBranch(self.point_feat_out_channels, 1),
            nn.Sigmoid()
        )

        # CFFE
        if hasattr(self.pModel, "CFFEParam"):
            cffe_cfg = self.pModel.CFFEParam
            # BEV network
            self.point2bev_cffe = get_module(cffe_cfg.BEVParam.P2VParam)
            self.bev_cffe = get_module(cffe_cfg.BEVParam, in_channel_list=(self.bev_net.out_channels, self.point_feat_out_channels), out_channel=self.bev_net.out_channels)
            self.bev2point_cffe = get_module(cffe_cfg.BEVParam.V2PParam)

            # RV network
            self.point2rv_cffe = get_module(cffe_cfg.RVParam.P2VParam)
            self.rv_cffe = get_module(cffe_cfg.RVParam, in_channel_list=(self.rv_net.out_channels, self.point_feat_out_channels), out_channel=self.rv_net.out_channels)
            self.rv2point_cffe = get_module(cffe_cfg.RVParam.V2PParam)

            # sem branch
            point_fusion_channels = (bev_base_channels[0], self.bev_net.out_channels, self.rv_net.out_channels)
            self.point_fusion_sem_cffe = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
            self.pred_layer_sem_cffe = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

            # ins branch
            self.point_fusion_ins_cffe = get_module(fusion_cfg, in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
            self.pred_layer_offset_cffe = backbone.PredBranch(self.point_feat_out_channels, 3)
            self.pred_layer_hmap_cffe = nn.Sequential(
                backbone.PredBranch(self.point_feat_out_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            pcds_xyzi (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        Output:
            pred_sem (BS, C, N, 1)
            pred_offset (BS, N, 3)
            pred_hmap (BS, N, 1)
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        point_feat_tmp = self.point_pre(pcds_xyzi)

        # BEV network
        bev_input = self.point2bev(point_feat_tmp, pcds_coord_wl)
        bev_feat_sem, bev_feat_ins = self.bev_net(bev_input)

        point_bev_sem = self.bev2point(bev_feat_sem, pcds_coord_wl)
        point_bev_ins = self.bev2point(bev_feat_ins, pcds_coord_wl)

        # RV network
        rv_input = self.point2rv(point_feat_tmp, pcds_sphere_coord)
        rv_feat_sem, rv_feat_ins = self.rv_net(rv_input)

        point_rv_sem = self.rv2point(rv_feat_sem, pcds_sphere_coord)
        point_rv_ins = self.rv2point(rv_feat_ins, pcds_sphere_coord)

        # stage0
        # sem branch
        point_feat_sem = self.point_fusion_sem(point_feat_tmp, point_bev_sem, point_rv_sem)
        pred_sem = self.pred_layer_sem(point_feat_sem).float()

        # ins branch
        point_feat_ins = self.point_fusion_ins(point_feat_tmp, point_bev_ins, point_rv_ins)
        pred_offset = self.pred_layer_offset(point_feat_ins).float().squeeze(-1).transpose(1, 2).contiguous()
        pred_hmap = self.pred_layer_hmap(point_feat_ins).float().squeeze(1)

        preds_list = [(pred_sem, pred_offset, pred_hmap)]

        if hasattr(self.pModel, "CFFEParam"):
            # CFFE
            pred_offset_high_conf = pred_offset.detach() * (pred_hmap > self.pModel.score_thresh).float()
            # reprojection
            pcds_coord_wl_reproj, pcds_sphere_coord_reproj = common_utils.reproj_with_offset(pcds_xyzi, pred_offset_high_conf,\
                self.pModel.Voxel, self.dx, self.dy, self.phi_range_radian, self.theta_range_radian, self.dphi, self.dtheta)
            
            # BEV network
            bev_cfg_feat = self.point2bev_cffe(point_feat_sem, pcds_coord_wl_reproj)
            bev_feat_sem_final, bev_feat_ins_final = self.bev_cffe(bev_feat_sem, bev_cfg_feat)

            point_bev_sem_cffe = self.bev2point_cffe(bev_feat_sem_final, pcds_coord_wl)
            point_bev_ins_cffe = self.bev2point_cffe(bev_feat_ins_final, pcds_coord_wl)

            # RV network
            rv_cfg_feat = self.point2rv_cffe(point_feat_sem, pcds_sphere_coord_reproj)
            rv_feat_sem_final, rv_feat_ins_final = self.rv_cffe(rv_feat_sem, rv_cfg_feat)

            point_rv_sem_cffe = self.rv2point_cffe(rv_feat_sem_final, pcds_sphere_coord)
            point_rv_ins_cffe = self.rv2point_cffe(rv_feat_ins_final, pcds_sphere_coord)

            # sem branch
            point_feat_sem_cffe = self.point_fusion_sem_cffe(point_feat_tmp, point_bev_sem_cffe, point_rv_sem_cffe)
            pred_sem_cffe = self.pred_layer_sem_cffe(point_feat_sem_cffe).float()

            # ins branch
            point_feat_ins_cffe = self.point_fusion_ins_cffe(point_feat_tmp, point_bev_ins_cffe, point_rv_ins_cffe)
            pred_offset_cffe = self.pred_layer_offset_cffe(point_feat_ins_cffe).float().squeeze(-1).transpose(1, 2).contiguous()
            pred_hmap_cffe = self.pred_layer_hmap_cffe(point_feat_ins_cffe).float().squeeze(1)

            preds_list.append((pred_sem_cffe, pred_offset_cffe, pred_hmap_cffe))
        
        return preds_list