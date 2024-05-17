import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

import numpy as np

import pan_lib.cuda_kernel
import copy

import pdb


# pcds_feat, (BS, C, N, 1)
# pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
# voxel_out, (BS, C, D1, D2, ..., Dn)
class VoxelMaxPoolFunction(Function):
    @staticmethod
    def forward(ctx, pcds_feat, pcds_ind, output_size, scale_rate):
        assert(pcds_ind.dtype == torch.float)
        assert(pcds_feat.dim() == 4)
        assert(pcds_ind.dim() == 4)
        assert(pcds_feat.size(2) == pcds_ind.size(1))
        assert(pcds_ind.size(2) == len(output_size))
        assert(pcds_ind.size(2) == len(scale_rate))

        voxel_out_shape = [pcds_feat.size(0), pcds_feat.size(1)] + list(output_size)
        voxel_out = torch.zeros(voxel_out_shape, dtype=pcds_feat.dtype, device=pcds_feat.device)

        voxel_out_size_pt = torch.LongTensor(voxel_out_shape).to(pcds_feat.device)
        voxel_out_stride_pt = torch.LongTensor(voxel_out.stride()).to(pcds_feat.device)
        output_size_pt = voxel_out_size_pt[2:]
        scale_rate_pt = torch.FloatTensor(scale_rate).to(pcds_feat.device)

        ctx.use_cuda = pcds_feat.is_cuda
        if ctx.use_cuda:
            pan_lib.cuda_kernel.voxel_maxpooling_forward(pcds_feat, pcds_ind, voxel_out,
            voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
        else:
            raise NotImplementedError
        
        ctx.input_shape = pcds_feat.shape
        ctx.save_for_backward(pcds_feat, pcds_ind, voxel_out, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
        return voxel_out
    
    @staticmethod
    def backward(ctx, grad_voxel_out):
        pcds_feat, pcds_ind, voxel_out, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_voxel_out = grad_voxel_out.contiguous()
            grad_pcds_feat = torch.zeros(ctx.input_shape, dtype=grad_voxel_out.dtype, device=grad_voxel_out.device)
            if ctx.use_cuda:
                pan_lib.cuda_kernel.voxel_maxpooling_backward(pcds_feat, pcds_ind, voxel_out,
                grad_pcds_feat, grad_voxel_out, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt, scale_rate_pt)
            else:
                raise NotImplementedError
            
            return grad_pcds_feat, None, None, None
        else:
            return None, None, None, None


def VoxelMaxPool(pcds_feat, pcds_ind, output_size, scale_rate):
    return VoxelMaxPoolFunction.apply(pcds_feat, pcds_ind, output_size, scale_rate)


# forward
# grid_in, (BS, C, H, W)
# pcds_ind,(BS, N, 2, 1), 2 -> h, w
# pcds_feat, (BS, C, N, 1)
class Grid2PointFunction(Function):
    @staticmethod
    def forward(ctx, grid_in, pcds_ind, scale_rate):
        assert(pcds_ind.dtype == torch.float)
        assert(grid_in.dim() == 4)
        assert(pcds_ind.dim() == 4)

        assert(pcds_ind.size(2) == 2)
        assert(len(scale_rate) == 2)

        pcds_feat = torch.zeros([grid_in.size(0), grid_in.size(1), pcds_ind.size(1), 1], dtype=grid_in.dtype, device=grid_in.device)
        
        grid_in_size_pt = torch.LongTensor(list(grid_in.shape)).to(grid_in.device)
        grid_in_stride_pt = torch.LongTensor(list(grid_in.stride())).to(grid_in.device)
        scale_rate_pt = torch.FloatTensor(scale_rate).to(grid_in.device)

        ctx.use_cuda = grid_in.is_cuda
        if ctx.use_cuda:
            pan_lib.cuda_kernel.grid2point_forward(pcds_feat, pcds_ind, grid_in,
            grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
        else:
            raise NotImplementedError
        
        ctx.input_shape = grid_in.shape
        ctx.save_for_backward(pcds_ind, grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
        return pcds_feat
    
    @staticmethod
    def backward(ctx, grad_pcds_feat):
        pcds_ind, grid_in_size_pt, grid_in_stride_pt, scale_rate_pt = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_pcds_feat = grad_pcds_feat.contiguous()
            grad_grid_in = torch.zeros(ctx.input_shape, dtype=grad_pcds_feat.dtype, device=grad_pcds_feat.device)
            if ctx.use_cuda:
                pan_lib.cuda_kernel.grid2point_backward(pcds_ind, grad_pcds_feat, grad_grid_in,
                grid_in_size_pt, grid_in_stride_pt, scale_rate_pt)
            else:
                raise NotImplementedError
            
            return grad_grid_in, None, None
        else:
            return None, None, None


def Grid2Point(grid_in, pcds_ind, scale_rate):
    return Grid2PointFunction.apply(grid_in, pcds_ind, scale_rate)


# voxel_sum
# pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
# voxel_out, (BS, D1, D2, ..., Dn)
class VoxelSumFunction(Function):
    @staticmethod
    def forward(ctx, pcds_ind, output_size):
        assert(pcds_ind.dim() == 4)
        assert(pcds_ind.size(3) == 1)
        assert(pcds_ind.size(2) == len(output_size))
        assert(pcds_ind.dtype == torch.long)
        
        voxel_out_shape = [pcds_ind.size(0)] + list(output_size)
        voxel_out = torch.zeros(voxel_out_shape, dtype=pcds_ind.dtype, device=pcds_ind.device)
        
        voxel_out_size_pt = torch.LongTensor(voxel_out_shape).to(pcds_ind.device)
        voxel_out_stride_pt = torch.LongTensor(voxel_out.stride()).to(pcds_ind.device)
        output_size_pt = voxel_out_size_pt[1:]
        
        pan_lib.cuda_kernel.voxel_sum_gpu(pcds_ind, voxel_out, voxel_out_size_pt, voxel_out_stride_pt, output_size_pt)
        return voxel_out
    
    @staticmethod
    def backward(ctx, grad_voxel_out):
        return None, None


def VoxelSum(pcds_ind, output_size):
    return VoxelSumFunction.apply(pcds_ind, output_size)


# voxel_query
# voxel_in, (BS, C, D1, D2, ..., Dn)
# pcds_ind,(BS, N, D, 1), D -> d1, d2, ..., dn
# pcds_feat, (BS, C, N, 1)
class VoxelQueryFunction(Function):
    @staticmethod
    def forward(ctx, voxel_in, pcds_ind):
        assert(voxel_in.dim() == (pcds_ind.size(2) + 2))
        assert(pcds_ind.dim() == 4)
        assert(pcds_ind.size(3) == 1)
        assert(pcds_ind.dtype == torch.long)
        
        pcds_feat = torch.zeros([voxel_in.size(0), voxel_in.size(1), pcds_ind.size(1), 1], dtype=voxel_in.dtype, device=voxel_in.device)
        
        voxel_in_size_pt = torch.LongTensor(list(voxel_in.shape)).to(voxel_in.device)
        voxel_in_stride_pt = torch.LongTensor(list(voxel_in.stride())).to(voxel_in.device)
        
        pan_lib.cuda_kernel.voxel_query_gpu(pcds_feat, pcds_ind, voxel_in, voxel_in_size_pt, voxel_in_stride_pt)
        return pcds_feat
    
    @staticmethod
    def backward(ctx, grad_pcds_feat):
        return None, None


def VoxelQuery(voxel_in, pcds_ind):
    return VoxelQueryFunction.apply(voxel_in, pcds_ind)


class PointVoteNMS(nn.Module):
    def __init__(self, point_nms_dic):
        super(PointVoteNMS, self).__init__()
        self.point_nms_dic = point_nms_dic
        self.K = self.point_nms_dic['K']
        self.dist_thresh = float(self.point_nms_dic['dist_thresh'])
        self.vote_thresh = float(self.point_nms_dic['vote_thresh'])
        self.score_thresh = float(self.point_nms_dic['score_thresh'])
        print('pv nms params: ', self.point_nms_dic)
    
    def forward(self, pcds_xyz, pred_sem, pred_offset, pred_hmap):
        '''
        Input:
            pcds_xyz, (N, C), C -> (x, y, z, intensity, ...)
            pred_sem, (N, class_num)
            pred_offset, (N, 3)
            pred_hmap, (N)
        Output:
            obj_center, (K, 3)
            pred_panoptic, (N)
        '''
        assert(pcds_xyz.dim() == 2)
        assert(pred_sem.dim() == 2)
        assert(pred_hmap.dim() == 1)
        assert(pred_offset.dim() == 2)

        N = pcds_xyz.shape[0]
        assert(pred_sem.shape[0] == N)
        assert(pred_hmap.shape[0] == N)
        assert(pred_offset.shape[0] == N)
        assert(pred_offset.shape[1] == 3)

        pred_panoptic = pred_sem.argmax(dim=1)

        obj_center = torch.zeros((self.K, 3), dtype=pcds_xyz.dtype, device=pcds_xyz.device)
        pcds_xyz_score = torch.cat((pcds_xyz + pred_offset, pred_hmap.unsqueeze(1)), dim=1)

        # get foreground and high-confidence mask
        valid_fg_mask = (pred_panoptic >= 1) * (pred_panoptic <= 8)
        valid_high_score_mask = (pcds_xyz_score[:, -1] > self.score_thresh) * valid_fg_mask
        pcds_fg = pcds_xyz_score[valid_high_score_mask]

        # gpu process
        N_fg = pcds_fg.shape[0]
        col_blocks = (N_fg + 64 - 1) // 64
        if N_fg > 1:
            _, indices = pcds_fg[:, -1].sort(dim=0, descending=True)
            pcds_fg_order = pcds_fg[indices]

            matching_mat = torch.zeros((N_fg, col_blocks), dtype=torch.long, device=pcds_xyz.device)
            matching_mat_vote = torch.zeros((N_fg, col_blocks), dtype=torch.long, device=pcds_xyz.device)
            remv = torch.zeros((col_blocks,), dtype=torch.long, device=pcds_xyz.device)
            keep = torch.zeros((self.K,), dtype=torch.long, device=pcds_xyz.device)

            pan_lib.cuda_kernel.vote_nms_fast_gpu(pcds_fg_order, obj_center, matching_mat, matching_mat_vote, remv, keep, self.dist_thresh, self.vote_thresh)
            valid_center_mask = obj_center.abs().sum(dim=1) > 0
            obj_center_valid = obj_center[valid_center_mask]

            # assign instance label
            dist = (obj_center_valid.unsqueeze(0) - pcds_xyz_score[valid_fg_mask][:, :3].unsqueeze(1)).norm(dim=-1) #(N1, K1)
            instance_id = dist.argmin(dim=1) + 1

            # majarity voting
            sem_ins_index = torch.stack((pred_panoptic[valid_fg_mask], instance_id), dim=1).view(1, -1, 2, 1) #(1, N1, 2, 1)
            pred_sem_ins_vote = VoxelSum(pcds_ind=sem_ins_index, output_size=(100, int(instance_id.max()) + 1)) #(1, class_num, gt_obj_max)
            _, pred_sem_ins_vote_id = pred_sem_ins_vote.max(dim=1, keepdim=True) #(1, 1, gt_obj_max)
            pred_sem_vote = VoxelQuery(voxel_in=pred_sem_ins_vote_id, pcds_ind=instance_id.view(1, -1, 1, 1)).view(-1) #(N1)

            assert(int(pred_sem_vote.max()) <= 8)
            assert(int(pred_sem_vote.min()) >= 1)
            # add instance_id
            pred_panoptic[valid_fg_mask] = pred_sem_vote
            pred_panoptic[valid_fg_mask] += instance_id << 16
            return obj_center_valid, pred_panoptic
        else:
            return None, pred_panoptic