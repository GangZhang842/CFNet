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


class MVFuse(nn.Module):
    def __init__(self, pModel, bev_channels, rv_channels):
        super(MVFuse, self).__init__()
        self.pModel = pModel
        self.bev_channels = bev_channels
        self.rv_channels = rv_channels

        # build network
        bev_context_layer = copy.deepcopy(self.pModel.BEVParam.context_layers)
        bev_grid2point = self.pModel.BEVParam.bev_grid2point

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_grid2point = self.pModel.RVParam.rv_grid2point

        fusion_mode = self.pModel.fusion_mode

        # sem branch
        self.bev_grid2point = get_module(bev_grid2point)
        self.rv_grid2point = get_module(rv_grid2point)

        point_fusion_channels_sem = (bev_context_layer[0], self.bev_channels, self.rv_channels)
        self.point_post_sem = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels_sem, out_channel=self.pModel.point_feat_out_channels)
        self.pred_layer_sem = backbone.PredBranch(self.pModel.point_feat_out_channels, self.pModel.class_num)

        # ins branch
        point_fusion_channels_ins = (bev_context_layer[0], self.bev_channels, self.rv_channels)
        self.point_post_ins = eval('backbone.{}'.format(fusion_mode))(in_channel_list=point_fusion_channels_ins, out_channel=self.pModel.point_feat_out_channels)
        self.pred_layer_offset = backbone.PredBranch(self.pModel.point_feat_out_channels, 3)
        self.pred_layer_hmap = nn.Sequential(
            backbone.PredBranch(self.pModel.point_feat_out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, point_feat_tmp, bev_feat_sem, bev_feat_ins, pc_coord_bev, rv_feat_sem, rv_feat_ins, pc_coord_rv):
        # sem branch
        point_bev_sem = self.bev_grid2point(bev_feat_sem, pc_coord_bev)
        point_rv_sem = self.rv_grid2point(rv_feat_sem, pc_coord_rv)
        point_feat_sem = self.point_post_sem(point_feat_tmp, point_bev_sem, point_rv_sem)
        pred_sem = self.pred_layer_sem(point_feat_sem).float()

        # ins branch
        point_bev_ins = self.bev_grid2point(bev_feat_ins, pc_coord_bev)
        point_rv_ins = self.rv_grid2point(rv_feat_ins, pc_coord_rv)
        point_feat_ins = self.point_post_ins(point_feat_tmp, point_bev_ins, point_rv_ins)
        pred_offset = self.pred_layer_offset(point_feat_ins).float().squeeze(-1).transpose(1, 2).contiguous()
        pred_hmap = self.pred_layer_hmap(point_feat_ins).float().squeeze(1)
        return point_feat_sem, pred_sem, pred_offset, pred_hmap


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

        rv_context_layer = copy.deepcopy(self.pModel.RVParam.context_layers)
        rv_layers = copy.deepcopy(self.pModel.RVParam.layers)
        rv_base_block = self.pModel.RVParam.base_block

        fusion_mode = self.pModel.fusion_mode
        double_branch = self.pModel.double_branch
        ce_fusion_mode = self.pModel.ce_fusion_mode

        # base network
        self.point_pre = backbone.PointNetStacker(7, bev_context_layer[0], pre_bn=True, stack_num=2)
        self.bev_net = bird_view.BEVNet(bev_base_block, bev_context_layer, bev_layers, double_branch)
        self.rv_net = range_view.RVNet(rv_base_block, rv_context_layer, rv_layers, double_branch)

        # mv fuse 0
        self.mv_fuse0 = MVFuse(self.pModel, self.bev_net.out_channels, self.rv_net.out_channels)
        self.bev_fuse0 = eval('common_utils.{}'.format(ce_fusion_mode))(self.bev_net.out_channels, self.point_feat_out_channels, self.bev_net.out_channels, double_branch)
        self.rv_fuse0 = eval('common_utils.{}'.format(ce_fusion_mode))(self.rv_net.out_channels, self.point_feat_out_channels, self.rv_net.out_channels, double_branch)

        # mv fuse 1
        self.mv_fuse1 = MVFuse(self.pModel, self.bev_net.out_channels, self.rv_net.out_channels)
    
    def forward_once(self, point_feat, pcds_coord, pcds_sphere_coord):
        '''
        Input:
            point_feat (BS, C, N, 1)
            pcds_coord (BS, N, 3, 1), 3 -> (x_quan, y_quan, z_quan)
            pcds_sphere_coord (BS, N, 2, 1), 2 -> (vertical_quan, horizon_quan)
        '''
        pcds_coord_wl = pcds_coord[:, :, :2].contiguous()
        point_feat_tmp = self.point_pre(point_feat)

        # range-view
        rv_input = common_utils.VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_sphere_coord, output_size=self.rv_shape, scale_rate=(1.0, 1.0))
        rv_feat_sem, rv_feat_ins = self.rv_net(rv_input)

        # bird-view
        bev_input = common_utils.VoxelMaxPool(pcds_feat=point_feat_tmp, pcds_ind=pcds_coord_wl, output_size=self.bev_wl_shape, scale_rate=(1.0, 1.0))
        bev_feat_sem, bev_feat_ins = self.bev_net(bev_input)

        # mv fuse0
        point_feat_sem0, pred_sem0, pred_offset0, pred_hmap0 =\
            self.mv_fuse0(point_feat_tmp, bev_feat_sem, bev_feat_ins, pcds_coord_wl, rv_feat_sem, rv_feat_ins, pcds_sphere_coord)
        
        valid_high_conf = (pred_hmap0 > self.point_nms_dic['score_thresh']).squeeze(-1) #(BS, N)
        pcds_xyzi_offset0 = point_feat.squeeze(-1).permute(0, 2, 1)[:, :, :3] + pred_offset0.detach()

        # reprojection
        pcds_coord_wl_reproj, pcds_sphere_coord_reproj = common_utils.reproj(pcds_xyzi_offset0,\
            self.pModel.Voxel, self.dx, self.dy, self.phi_range_radian, self.theta_range_radian, self.dphi, self.dtheta)
        pcds_coord_wl_reproj[~valid_high_conf] = -1
        pcds_sphere_coord_reproj[~valid_high_conf] = -1

        rv_reproj_feat = common_utils.VoxelMaxPool(pcds_feat=point_feat_sem0, pcds_ind=pcds_sphere_coord_reproj, output_size=tuple(rv_feat_sem.shape[2:]), scale_rate=self.pModel.RVParam.rv_grid2point['scale_rate'])
        bev_reproj_feat = common_utils.VoxelMaxPool(pcds_feat=point_feat_sem0, pcds_ind=pcds_coord_wl_reproj, output_size=tuple(bev_feat_sem.shape[2:]), scale_rate=self.pModel.BEVParam.bev_grid2point['scale_rate'])

        rv_feat_sem_final, rv_feat_ins_final = self.rv_fuse0(rv_feat_sem, rv_reproj_feat)
        bev_feat_sem_final, bev_feat_ins_final = self.bev_fuse0(bev_feat_sem, bev_reproj_feat)

        point_feat_sem1, pred_sem1, pred_offset1, pred_hmap1 =\
            self.mv_fuse1(point_feat_tmp, bev_feat_sem_final, bev_feat_ins_final, pcds_coord_wl, rv_feat_sem_final, rv_feat_ins_final, pcds_sphere_coord)
        
        return pred_sem0, pred_offset0, pred_hmap0, pred_sem1, pred_offset1, pred_hmap1
    
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
        pred_sem0, pred_offset0, pred_hmap0, pred_sem1, pred_offset1, pred_hmap1 = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        pred_sem0_raw, pred_offset0_raw, pred_hmap0_raw, pred_sem1_raw, pred_offset1_raw, pred_hmap1_raw = self.forward_once(pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw)

        # total loss
        loss_pano0 = self.panoptic_loss(pred_sem0, pred_offset0, pred_hmap0, pcds_offset, pcds_ins_label, pcds_sem_label)
        loss_pano1 = self.panoptic_loss(pred_sem1, pred_offset1, pred_hmap1, pcds_offset, pcds_ins_label, pcds_sem_label)

        loss_pano_raw0 = self.panoptic_loss(pred_sem0_raw, pred_offset0_raw, pred_hmap0_raw, pcds_offset_raw, pcds_ins_label, pcds_sem_label)
        loss_pano_raw1 = self.panoptic_loss(pred_sem1_raw, pred_offset1_raw, pred_hmap1_raw, pcds_offset_raw, pcds_ins_label, pcds_sem_label)

        loss_consist0 = self.consistency_loss_l1(pred_sem0, pred_sem0_raw)
        loss_consist1 = self.consistency_loss_l1(pred_sem1, pred_sem1_raw)

        loss = 0.5 * (loss_pano0 + loss_pano1 + loss_pano_raw0 + loss_pano_raw1) + loss_consist0 + loss_consist1
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
        pred_sem0, pred_offset0, pred_hmap0, pred_sem1, pred_offset1, pred_hmap1 = self.forward_once(pcds_xyzi, pcds_coord, pcds_sphere_coord)
        return pred_sem1, pred_offset1, pred_hmap1