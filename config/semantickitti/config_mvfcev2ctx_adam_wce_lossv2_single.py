import numpy as np
import os
from functools import reduce


def get_config():
    class General:
        name = reduce(lambda x,y:os.path.join(x,y), __name__.rsplit("/")[-1].split('.')[1:])
        batch_size_per_gpu = 2

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
            type = 'datasets.data.DataloadTrain'
            num_workers = 8
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            center_type = General.center_type
            class CopyPasteAug:
                is_use = True
                ObjBackDir = './data/object_bank_semkitti'
                paste_max_obj_num = 20
            class AugParam:
                noise_mean = 0
                noise_std = 0.0001
                theta_range = (-180.0, 180.0)
                shift_range = ((-1, 1), (-1, 1), (-0.4, 0.4))
                size_range = (0.95, 1.05)
        
        class Val:
            type = 'datasets.data.DataloadVal'
            num_workers = 8
            frame_point_num = 130000
            SeqDir = General.SeqDir
            Voxel = General.Voxel
            center_type = General.center_type
    
    class ModelParam:
        type = "models.cfnet.CFNet"
        runner_type = "models.model_runner.ModelRunnerSemKITTI"
        score_thresh = 0.2
        category_list = General.category_list
        Voxel = General.Voxel
        class_num = len(General.category_list) + 1
        point_nms_dic = General.point_nms_dic
        point_feat_out_channels = 64

        class BEVParam:
            type = 'models.networks.fcn_2d.FCN2D'
            base_block = 'models.networks.backbone.BasicBlock'
            fpn_block = 'models.networks.fpn.AttMerge'
            base_channels = [64, 40, 80, 160]
            base_layers = [2, 3, 4]
            base_strides = [2, 2, 2]
            double_branch = True
            class P2VParam:
                type = 'models.networks.point2voxel.VMPModule'
                output_size = General.Voxel.bev_shape[:2]
                scale_rate = [1, 1]
            
            class V2PParam:
                type = 'models.networks.voxel2point.G2PModule'
                scale_rate = (0.5, 0.5)
        
        class RVParam:
            type = 'models.networks.fcn_2d.FCN2D'
            base_block = 'models.networks.backbone.BasicBlock'
            fpn_block = 'models.networks.fpn.AttMerge'
            base_channels = [64, 40, 80, 160]
            base_layers = [2, 3, 4]
            base_strides = ((1, 2), 2, 2)
            double_branch = True
            class P2VParam:
                type = 'models.networks.point2voxel.VMPModule'
                output_size = General.Voxel.rv_shape
                scale_rate = [1, 1]
            
            class V2PParam:
                type = 'models.networks.voxel2point.G2PModule'
                scale_rate = (1.0, 0.5)
        
        class CFFEParam:
            class BEVParam:
                type = 'models.networks.fusion_module.CatFusionCtx'
                double_branch = True
                class P2VParam:
                    type = 'models.networks.point2voxel.VMPModule'
                    output_size = [General.Voxel.bev_shape[0] // 2, General.Voxel.bev_shape[1] // 2]
                    scale_rate = [0.5, 0.5]
                
                class V2PParam:
                    type = 'models.networks.voxel2point.G2PModule'
                    scale_rate = (0.5, 0.5)
            
            class RVParam:
                type = 'models.networks.fusion_module.CatFusionCtx'
                double_branch = True
                class P2VParam:
                    type = 'models.networks.point2voxel.VMPModule'
                    output_size = [General.Voxel.rv_shape[0], General.Voxel.rv_shape[1] // 2]
                    scale_rate = [1.0, 0.5]
                
                class V2PParam:
                    type = 'models.networks.voxel2point.G2PModule'
                    scale_rate = (1.0, 0.5)
        
        class FusionParam:
            type = 'models.networks.fusion_module.PointCatFusion'
        
        class LossParam:
            type = 'models.loss.criterion.PanopticLossv2_single'
            loss_seg_dic = dict(type='wce')
            loss_ins_dic = dict(top_ratio=0.1, top_weight=3.0)
            sigma = np.sqrt(-1 * General.point_nms_dic['dist_thresh'] * General.point_nms_dic['dist_thresh'] / (2 * np.log(General.point_nms_dic['score_thresh'])))
            ce_weight = 1.0
            lovasz_weight = 3.0
            center_weight = 1.0
            offset_weight = 2.0
            ignore_index = 0
        
        class optimizer:
            type = "adam"
            base_lr = 0.003
            momentum = 0.9
            nesterov = True
            wd = 1e-3
        
        class scheduler:
            type = "OneCycle"
            max_epochs = 48
            pct_start = 0.3
            final_lr = 1e-6
    
    return General, DatasetParam, ModelParam