import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function

import pytorch_lib

from .lovasz_losses import lovasz_softmax

import yaml

import numpy as np
import pdb


def get_ohem_loss(loss_mat, valid_mask=None, top_ratio=0, top_weight=1):
    loss_mat_valid = None
    valid_num = None
    topk_num = None
    if valid_mask is not None:
        loss_mat_valid = (loss_mat * valid_mask).view(-1)
        valid_num = int(valid_mask.sum())
        topk_num = int(valid_num * top_ratio)
    else:
        loss_mat_valid = loss_mat.view(-1)
        valid_num = loss_mat_valid.shape[0]
        topk_num = int(valid_num * top_ratio)
    
    loss_total = loss_mat_valid.sum() / (valid_num + 1e-12)
    if topk_num == 0:
        return loss_total
    else:
        loss_topk = torch.topk(loss_mat_valid, k=topk_num, dim=0, largest=True, sorted=False)[0]
        loss_total = loss_total + top_weight * loss_topk.mean()
        return loss_total


# define the online semi-hard examples mining binary cross entropy
class CE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0, ignore_index=-1):
        super(CE_OHEM, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
        self.ignore_index = ignore_index

        self.loss_func = nn.CrossEntropyLoss(reduce=False, ignore_index=self.ignore_index)
    
    def forward(self, pred, gt):
        #pdb.set_trace()
        loss_mat = self.loss_func(pred, gt.long())
        valid_mask = (gt != self.ignore_index).float()

        loss_result = get_ohem_loss(loss_mat, valid_mask, top_ratio=self.top_ratio, top_weight=self.top_weight)
        return loss_result


# define the online semi-hard examples mining binary cross entropy
class BCE_OHEM(nn.Module):
    def __init__(self, top_ratio=0.3, top_weight=1.0):
        super(BCE_OHEM, self).__init__()
        self.top_ratio = top_ratio
        self.top_weight = top_weight
    
    def forward(self, pred, gt, valid_mask=None):
        #pdb.set_trace()
        # loss_mat = F.binary_cross_entropy(pred, gt, reduce=False)
        loss_mat = -1 * (gt * torch.log(pred + 1e-12) + (1 - gt) * torch.log(1 - pred + 1e-12))
        loss_result = get_ohem_loss(loss_mat, valid_mask, top_ratio=self.top_ratio, top_weight=self.top_weight)
        return loss_result


class PanopticLossv1(nn.Module):
    def __init__(self, ce_weight, lovasz_weight, center_weight, offset_weight, sigma, ignore_index, loss_seg_dic, loss_ins_dic):
        super(PanopticLossv1, self).__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.sigma = sigma
        self.ignore_index = ignore_index
        loss_seg_dic = loss_seg_dic
        loss_ins_dic = loss_ins_dic

        # seg loss
        self.criterion_seg = None
        if loss_seg_dic['type'] == 'ce':
            self.criterion_seg = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'ohem':
            self.criterion_seg = CE_OHEM(top_ratio=loss_seg_dic['top_ratio'], top_weight=loss_seg_dic['top_weight'], ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'wce':
            class_weight = None
            if 'class_weight' in loss_seg_dic:
                class_weight = torch.FloatTensor(loss_seg_dic['class_weight'])
            else:
                with open('datasets/semantic-kitti.yaml', 'r') as f:
                    task_cfg = yaml.safe_load(f)
                    content = torch.zeros(len(task_cfg['learning_ignore']), dtype=torch.float32)
                    for cl, freq in task_cfg["content"].items():
                        x_cl = task_cfg['learning_map'][cl]
                        content[x_cl] += freq
                    
                    class_weight = 1 / (content + 0.001)
                    class_weight[self.ignore_index] = 0
            
            print("Class weights: ", class_weight)
            self.criterion_seg = nn.CrossEntropyLoss(weight=class_weight.cuda())
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

        # ins loss
        self.center_loss = BCE_OHEM(top_ratio=loss_ins_dic['top_ratio'], top_weight=loss_ins_dic['top_weight'])
    
    def forward(self, pred_sem, pred_offset, pred_hmap, gt_offset, gt_ins_label, gt_sem):
        '''
        Input:
            pred_sem (BS, C, N, 1)
            pred_offset, gt_offset (BS, N, 3)
            pred_hmap (BS, N, 1)
            gt_ins_label, gt_sem (BS, N, 1)
        '''
        # seg loss
        loss_ce = self.criterion_seg(pred_sem, gt_sem)
        loss_lovasz = lovasz_softmax(pred_sem, gt_sem, ignore=self.ignore_index)

        # ins loss
        valid_mask = (gt_sem != self.ignore_index)
        fg_mask = (gt_ins_label >= 0)

        valid_num = int(valid_mask.float().sum()) + 1e-12
        fg_num = int(fg_mask.float().sum()) + 1e-12

        loss_point = (pred_offset - gt_offset).pow(2).sum(dim=2, keepdim=True).sqrt() #(BS, N, 1)
        gt_hmap = torch.exp(-1 * gt_offset.pow(2).sum(dim=2, keepdim=True) / (2 * self.sigma * self.sigma)) * fg_mask.float() #(BS, N, 1)

        loss_offset = (loss_point * fg_mask.float()).sum() / fg_num
        loss_center = self.center_loss(pred_hmap[valid_mask], gt_hmap[valid_mask])

        loss_total = self.ce_weight * loss_ce + self.lovasz_weight * loss_lovasz + self.center_weight * loss_center + self.offset_weight * loss_offset
        return loss_total


class PanopticLossv2(nn.Module):
    def __init__(self, ce_weight, lovasz_weight, center_weight, offset_weight, sigma, ignore_index, loss_seg_dic, loss_ins_dic):
        super(PanopticLossv2, self).__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.sigma = sigma
        self.ignore_index = ignore_index
        loss_seg_dic = loss_seg_dic
        loss_ins_dic = loss_ins_dic

        # seg loss
        self.criterion_seg = None
        if loss_seg_dic['type'] == 'ce':
            self.criterion_seg = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'ohem':
            self.criterion_seg = CE_OHEM(top_ratio=loss_seg_dic['top_ratio'], top_weight=loss_seg_dic['top_weight'], ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'wce':
            class_weight = None
            if 'class_weight' in loss_seg_dic:
                class_weight = torch.FloatTensor(loss_seg_dic['class_weight'])
            else:
                with open('datasets/semantic-kitti.yaml', 'r') as f:
                    task_cfg = yaml.safe_load(f)
                    content = torch.zeros(len(task_cfg['learning_ignore']), dtype=torch.float32)
                    for cl, freq in task_cfg["content"].items():
                        x_cl = task_cfg['learning_map'][cl]
                        content[x_cl] += freq
                    
                    class_weight = 1 / (content + 0.001)
                    class_weight[self.ignore_index] = 0
            
            print("Class weights: ", class_weight)
            self.criterion_seg = nn.CrossEntropyLoss(weight=class_weight.cuda())
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')
        
        # ins loss
        self.center_loss = BCE_OHEM(top_ratio=loss_ins_dic['top_ratio'], top_weight=loss_ins_dic['top_weight'])
    
    def forward(self, pred_sem, pred_offset, pred_hmap, gt_offset, gt_ins_label, gt_sem):
        '''
        Input:
            pred_sem (BS, C, N, 1)
            pred_offset, gt_offset (BS, N, 3)
            pred_hmap (BS, N, 1)
            gt_ins_label, gt_sem (BS, N, 1)
        '''
        # seg loss
        loss_ce = self.criterion_seg(pred_sem, gt_sem)
        loss_lovasz = lovasz_softmax(pred_sem, gt_sem, ignore=self.ignore_index)

        # ins loss
        valid_mask = (gt_sem != self.ignore_index)
        fg_mask = (gt_ins_label >= 0)

        valid_num = int(valid_mask.float().sum()) + 1e-12
        fg_num = int(fg_mask.float().sum()) + 1e-12

        loss_point = (pred_offset - gt_offset).pow(2).sum(dim=2, keepdim=True).sqrt() #(BS, N, 1)
        gt_hmap = torch.exp(-1 * loss_point.detach().pow(2) / (2 * self.sigma * self.sigma)) * fg_mask.float() #(BS, N, 1)

        loss_offset = (loss_point * fg_mask.float()).sum() / fg_num
        loss_center = self.center_loss(pred_hmap[valid_mask], gt_hmap[valid_mask])

        loss_total = self.ce_weight * loss_ce + self.lovasz_weight * loss_lovasz + self.center_weight * loss_center + self.offset_weight * loss_offset
        return loss_total


class PanopticLossv2_single(nn.Module):
    def __init__(self, ce_weight, lovasz_weight, center_weight, offset_weight, sigma, ignore_index, loss_seg_dic, loss_ins_dic):
        super(PanopticLossv2_single, self).__init__()
        self.ce_weight = ce_weight
        self.lovasz_weight = lovasz_weight
        self.center_weight = center_weight
        self.offset_weight = offset_weight
        self.sigma = sigma
        self.ignore_index = ignore_index
        loss_seg_dic = loss_seg_dic
        loss_ins_dic = loss_ins_dic

        # seg loss
        self.criterion_seg = None
        if loss_seg_dic['type'] == 'ce':
            self.criterion_seg = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'ohem':
            self.criterion_seg = CE_OHEM(top_ratio=loss_seg_dic['top_ratio'], top_weight=loss_seg_dic['top_weight'], ignore_index=self.ignore_index)
        elif loss_seg_dic['type'] == 'wce':
            class_weight = None
            if 'class_weight' in loss_seg_dic:
                class_weight = torch.FloatTensor(loss_seg_dic['class_weight'])
            else:
                with open('datasets/semantic-kitti.yaml', 'r') as f:
                    task_cfg = yaml.safe_load(f)
                    content = torch.zeros(len(task_cfg['learning_ignore']), dtype=torch.float32)
                    for cl, freq in task_cfg["content"].items():
                        x_cl = task_cfg['learning_map'][cl]
                        content[x_cl] += freq
                    
                    class_weight = 1 / (content + 0.001)
                    class_weight[self.ignore_index] = 0
            
            print("Class weights: ", class_weight)
            self.criterion_seg = nn.CrossEntropyLoss(weight=class_weight.cuda())
        else:
            raise Exception('loss_mode must in ["ce", "wce", "ohem"]')

        # ins loss
        self.center_loss = BCE_OHEM(top_ratio=loss_ins_dic['top_ratio'], top_weight=loss_ins_dic['top_weight'])
    
    def get_fg_pc_offset_weight(self, gt_ins_label):
        BS = gt_ins_label.shape[0]
        N = gt_ins_label.shape[1]

        if gt_ins_label.max() >= 0:
            per_obj_num = pytorch_lib.VoxelSum(pcds_ind=gt_ins_label.view(BS, N, 1, 1), output_size=(int(gt_ins_label.max()) + 1,)).float().unsqueeze(1) #(BS, 1, K)
            obj_sum = int((per_obj_num > 0).sum())
            per_obj_weight = 1 / (per_obj_num * obj_sum + 1e-12) #(BS, 1, K)

            pc_offset_weight = pytorch_lib.VoxelQuery(voxel_in=per_obj_weight, pcds_ind=gt_ins_label.view(BS, N, 1, 1)).squeeze(1) #(BS, N, 1)
            return pc_offset_weight
        else:
            pc_offset_weight = torch.zeros_like(gt_ins_label).float()
            return pc_offset_weight
    
    def forward(self, pred_sem, pred_offset, pred_hmap, gt_offset, gt_ins_label, gt_sem):
        '''
        Input:
            pred_sem (BS, C, N, 1)
            pred_offset, gt_offset (BS, N, 3)
            pred_hmap (BS, N, 1)
            gt_ins_label, gt_sem (BS, N, 1)
        '''
        # seg loss
        loss_ce = self.criterion_seg(pred_sem, gt_sem)
        loss_lovasz = lovasz_softmax(pred_sem, gt_sem, ignore=self.ignore_index)

        # ins loss
        valid_mask = (gt_sem != self.ignore_index)
        fg_mask = (gt_ins_label >= 0)
        pc_offset_weight = self.get_fg_pc_offset_weight(gt_ins_label)

        valid_num = int(valid_mask.float().sum()) + 1e-12
        fg_num = int(fg_mask.float().sum()) + 1e-12

        loss_point = (pred_offset - gt_offset).pow(2).sum(dim=2, keepdim=True).sqrt() #(BS, N, 1)
        gt_hmap = torch.exp(-1 * loss_point.detach().pow(2) / (2 * self.sigma * self.sigma)) * fg_mask.float() #(BS, N, 1)

        loss_offset = (loss_point * pc_offset_weight).sum()
        loss_center = self.center_loss(pred_hmap[valid_mask], gt_hmap[valid_mask])

        loss_total = self.ce_weight * loss_ce + self.lovasz_weight * loss_lovasz + self.center_weight * loss_center + self.offset_weight * loss_offset
        return loss_total