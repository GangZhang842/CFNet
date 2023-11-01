import numpy as np


def get_config():
    class General:
        log_frequency = 100
        name = __name__.rsplit("/")[-1].rsplit(".")[-1]
        batch_size_per_gpu = 2
        fp16 = True

        SeqDir = './data/SemanticKITTI/dataset/sequences'
        category_list = ['car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                        'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk',
                        'terrain', 'pole', 'traffic-sign']
        
        center_type = 'axis'
        point_nms_dic = dict(K=100, score_thresh=0.2, dist_thresh=0.8, vote_thresh=0.4)
        class Voxel:
            RV_theta = (-25.0, 3.0)
            range_x = (-50.0, 50.0)
            range_y = (-50.0, 50.0)
            range_z = (-4.0, 2.0)

            bev_shape = (600, 600, 30)
            rv_shape = (64, 2048)
    
    class DatasetParam:
        class Train:
            data_src = 'data'
            num_workers = 4
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            center_type = General.center_type
            class CopyPasteAug:
                is_use = True
                ObjBackDir = './data/object_bank'
                paste_max_obj_num = 20
            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-1, 1), (-1, 1), (-0.4, 0.4))
                size_range = (0.95, 1.05)
        
        class Val:
            data_src = 'data'
            num_workers = 4
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            center_type = General.center_type
    
    class ModelParam:
        prefix = "mvf_ce_v2"
        Voxel = General.Voxel
        class_num = len(General.category_list) + 1
        point_nms_dic = General.point_nms_dic

        point_feat_out_channels = 64
        fusion_mode = 'CatFusion'
        double_branch = True
        ce_fusion_mode = 'CatFusionCtx'

        class BEVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 40, 80, 160)
            layers = (2, 3, 4)
            bev_grid2point = dict(type='BilinearSample', scale_rate=(0.5, 0.5))
        
        class RVParam:
            base_block = 'BasicBlock'
            context_layers = (64, 40, 80, 160)
            layers = (2, 3, 4)
            rv_grid2point = dict(type='BilinearSample', scale_rate=(1.0, 0.5))
        
        class LossParam:
            loss_type = 'PanopticLossv2_single'
            loss_seg_dic = dict(type='wce')
            loss_ins_dic = dict(top_ratio=0.1, top_weight=3.0)
            sigma = np.sqrt(-1 * General.point_nms_dic['dist_thresh'] * General.point_nms_dic['dist_thresh'] / (2 * np.log(General.point_nms_dic['score_thresh'])))
            ce_weight = 1.0
            lovasz_weight = 3.0
            center_weight = 1.0
            offset_weight = 2.0
            ignore_index = 0
        
        class pretrain:
            pretrain_epoch = 57
    
    class OptimizeParam:
        class optimizer:
            type = "sgd"
            base_lr = 0.02
            momentum = 0.9
            nesterov = True
            wd = 1e-3
        
        class schedule:
            type = "step"
            begin_epoch = 0
            end_epoch = 48
            pct_start = 0.01
            final_lr = 1e-6
            step = 10
            decay_factor = 0.1
    
    return General, DatasetParam, ModelParam, OptimizeParam