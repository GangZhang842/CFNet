import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks import backbone, bird_view, range_view
from networks.backbone import get_module

from utils.criterion import *
from . import common_utils

import yaml
import copy
import pdb


class AttNet(nn.Module):
    def __init__(self, pModel):
        super(AttNet, self).__init__()
        self.pModel = pModel

        self.bev_shape = list(pModel.Voxel.bev_shape)
        self.rv_shape = list(pModel.Voxel.rv_shape)
        self.bev_wl_shape = self.bev_shape[:2]

        self.dx = (pModel.Voxel.range_x[1] - pModel.Voxel.range_x[0]) / pModel.Voxel.bev_shape[0]
        self.dy = (pModel.Voxel.range_y[1] - pModel.Voxel.range_y[0]) / pModel.Voxel.bev_shape[1]

        self.phi_range_radian = (-np.pi, np.pi)
        self.theta_range_radian = (pModel.Voxel.RV_theta[0] * np.pi / 180.0, pModel.Voxel.RV_theta[1] * np.pi / 180.0)

        self.dphi = (self.phi_range_radian[1] - self.phi_range_radian[0]) / pModel.Voxel.rv_shape[1]
        self.dtheta = (self.theta_range_radian[1] - self.theta_range_radian[0]) / pModel.Voxel.rv_shape[0]

        self.point_feat_out_channels = pModel.point_feat_out_channels

        self.build_network()
        self.panoptic_loss = eval(self.pModel.LossParam.loss_type)(self.pModel.LossParam)
        self.point_nms_dic = pModel.point_nms_dic
    
    def build_network(self):
        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_layers = copy.deepcopy(self.pModel.BEVParam.layers)
        bev_base_block = self.pModel.BEVParam.base_block
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode
        double_branch = self.pModel.double_branch

        # base network
        self.point_pre = backbone.PointNetStacker(7, bev_context_layer[0], pre_bn=True, stack_num=2)
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers, double_branch)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, double_branch)

        # sem branch
        self.bev_grid2point = get_module(bev_grid2point)
        self.rv_grid2point = get_module(rv_grid2point)

        point_fusion_channels = (bev_context_layer[0], self.bev_net.out_channels, self.rv_net.out_channels)
        self.point_post_sem = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
        self.pred_layer_sem = backbone.PredBranch(self.point_feat_out_channels, self.pModel.class_num)

        # ins branch
        self.point_post_ins = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels, out_channel=self.point_feat_out_channels)
        self.pred_layer_offset = backbone.PredBranch(self.point_feat_out_channels, 3)
        self.pred_layer_hmap = nn.Sequential(
            backbone.PredBranch(self.point_feat_out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward_once(self, point_feat, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            point_feat (BS, C, N, 1)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        point_feat_tmp = self.point_pre(point_feat)

        #range-view
        rv_input = common_utils.VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_sphere_coord, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat_sem, rv_feat_ins = self.rv_net(rv_input)

        #bird-view
        bev_input = common_utils.VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord_wl, output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        bev_feat_sem, bev_feat_ins = self.bev_net(bev_input)

        # sem branch
        point_rv_sem = self.rv_grid2point(rv_feat_sem, pcds_sphere_coord)
        point_bev_sem = self.bev_grid2point(bev_feat_sem, pcds_coord_wl)
        point_feat_sem = self.point_post_sem(point_feat_tmp, point_bev_sem, point_rv_sem)
        pred_sem = self.pred_layer_sem(point_feat_sem).float()

        # ins branch
        point_rv_ins = self.rv_grid2point(rv_feat_ins, pcds_sphere_coord)
        point_bev_ins = self.bev_grid2point(bev_feat_ins, pcds_coord_wl)
        point_feat_ins = self.point_post_ins(point_feat_tmp, point_bev_ins, point_rv_ins)
        pred_offset = self.pred_layer_offset(point_feat_ins).float().squeeze(-1).transpose(1, 2).contiguous()
        pred_hmap = self.pred_layer_hmap(point_feat_ins).float().squeeze(1)

        return pred_sem, pred_offset, pred_hmap
    
    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
            pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)

        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss
    
    def forward(self, pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw):
        '''
        Input:
            pcds_xyzi, pcds_xyzi_raw (BS, 7, N, 1), 7 -> (x, y, z, intensity, dist, diff_x, diff_y)
            pcds_coord, pcds_coord_raw (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord, pcds_sphere_coord_raw (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
            pcds_sem_label, pcds_sem_label_raw, pcds_ins_label, pcds_ins_label_raw (BS, N, 1)
            pcds_offset, pcds_offset_raw (BS, N, 3)
        Output:
            loss_list
        '''
        pred_sem, pred_offset, pred_hmap = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        pred_sem_raw, pred_offset_raw, pred_hmap_raw = self.forward_once(pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)
        
        # total loss
        loss_pano = self.panoptic_loss(pred_sem, pred_offset, pred_hmap, pcds_offset, pcds_ins_label, pcds_sem_label)
        loss_pano_raw = self.panoptic_loss(pred_sem_raw, pred_offset_raw, pred_hmap_raw, pcds_offset_raw, pcds_ins_label, pcds_sem_label)
        loss_consist = self.consistency_loss_l1(pred_sem, pred_sem_raw)

        loss = 0.5 * (loss_pano + loss_pano_raw) + loss_consist
        return loss
    
    def infer(self, pcds_xyzi, pcds_coord, pcds_sphere_coord):
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
        pred_sem, pred_offset, pred_hmap = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        return pred_sem, pred_offset, pred_hmap