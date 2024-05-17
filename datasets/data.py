import torch

import PIL.Image as Im
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
import numpy.linalg as lg

import yaml
import random
import json
from . import utils, copy_paste
import os


def make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, Voxel):
    # make point feat
    x = pcds_xyzi[:, 0].copy()
    y = pcds_xyzi[:, 1].copy()
    z = pcds_xyzi[:, 2].copy()
    intensity = pcds_xyzi[:, 3].copy()

    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12

    # grid diff
    diff_x = pcds_coord[:, 0] - np.floor(pcds_coord[:, 0])
    diff_y = pcds_coord[:, 1] - np.floor(pcds_coord[:, 1])
    diff_z = pcds_coord[:, 2] - np.floor(pcds_coord[:, 2])

    # sphere diff
    phi_range_radian = (-np.pi, np.pi)
    theta_range_radian = (Voxel.RV_theta[0] * np.pi / 180.0, Voxel.RV_theta[1] * np.pi / 180.0)

    phi = phi_range_radian[1] - np.arctan2(x, y)
    theta = theta_range_radian[1] - np.arcsin(z / dist)

    diff_phi = pcds_sphere_coord[:, 0] - np.floor(pcds_sphere_coord[:, 0])
    diff_theta = pcds_sphere_coord[:, 1] - np.floor(pcds_sphere_coord[:, 1])

    point_feat = np.stack((x, y, z, intensity, dist, diff_x, diff_y), axis=-1)
    return point_feat


# define the class of dataloader
class DataloadTrain(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.safe_load(f)
        
        self.cp_aug = None
        if config.CopyPasteAug.is_use:
            self.cp_aug = copy_paste.CutPaste(config.CopyPasteAug.ObjBackDir, config.CopyPasteAug.paste_max_obj_num, road_idx=[9, 11])
        
        self.aug = utils.DataAugment(noise_mean=config.AugParam.noise_mean,
                        noise_std=config.AugParam.noise_std,
                        theta_range=config.AugParam.theta_range,
                        shift_range=config.AugParam.shift_range,
                        size_range=config.AugParam.size_range)

        self.aug_raw = utils.DataAugment(noise_mean=0,
                        noise_std=0,
                        theta_range=(0, 0),
                        shift_range=((0, 0), (0, 0), (0, 0)),
                        size_range=(1, 1))
        
        # add training data
        seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['train']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcds = os.path.join(fpath, 'velodyne')
            fpath_labels = os.path.join(fpath, 'labels')
            file_list_length = len([x for x in os.listdir(fpath_pcds) if x.endswith('.bin')])
            for fn_id in range(file_list_length):
                fname_pcds = os.path.join(fpath_pcds, f"{str(fn_id).rjust(6, '0')}.bin")
                fname_labels = os.path.join(fpath_labels, f"{str(fn_id).rjust(6, '0')}.label")
                self.flist.append((fname_pcds, fname_labels, seq_id, f"{str(fn_id).rjust(6, '0')}.bin"))
        
        print('Training Samples: ', len(self.flist))

    def form_batch(self, pcds_total):
        #augment pcds
        pcds_total = self.aug(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset

    def form_batch_raw(self, pcds_total):
        #augment pcds
        pcds_total = self.aug_raw(pcds_total)

        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset

    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]

        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32)
        pcds = pcds.reshape((-1, 4))

        pcds_label = np.fromfile(fname_labels, dtype=np.uint32)
        pcds_label = pcds_label.reshape((-1))

        sem_label = pcds_label & 0xFFFF
        inst_label = pcds_label >> 16

        pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
        pcds_ins_label = utils.gene_ins_label(pcds_label_use, inst_label)
        
        # copy-paste augmentation
        if self.cp_aug is not None:
            pcds, pcds_label_use, pcds_ins_label = self.cp_aug(pcds, pcds_label_use, pcds_ins_label)
        
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, pcds_label_use[:, np.newaxis], pcds_ins_label[:, np.newaxis]), axis=1)

        # resample
        choice = np.random.choice(pcds_total.shape[0], self.frame_point_num, replace=True)
        pcds_total = pcds_total[choice]

        # preprocess
        pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_total.copy())
        pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw = self.form_batch_raw(pcds_total.copy())
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset,\
            pcds_xyzi_raw, pcds_coord_raw, pcds_sphere_coord_raw, pcds_sem_label_raw, pcds_ins_label_raw, pcds_offset_raw, seq_id, fn

    def __len__(self):
        return len(self.flist)


# define the class of dataloader
class DataloadVal(Dataset):
    def __init__(self, config):
        self.flist = []
        self.config = config
        self.frame_point_num = config.frame_point_num
        self.Voxel = config.Voxel
        with open('datasets/semantic-kitti.yaml', 'r') as f:
            self.task_cfg = yaml.safe_load(f)
        
        seq_split = [str(i).rjust(2, '0') for i in self.task_cfg['split']['valid']]
        for seq_id in seq_split:
            fpath = os.path.join(config.SeqDir, seq_id)
            fpath_pcds = os.path.join(fpath, 'velodyne')
            fpath_labels = os.path.join(fpath, 'labels')
            file_list_length = len([x for x in os.listdir(fpath_pcds) if x.endswith('.bin')])
            for fn_id in range(file_list_length):
                fname_pcds = os.path.join(fpath_pcds, f"{str(fn_id).rjust(6, '0')}.bin")
                fname_labels = os.path.join(fpath_labels, f"{str(fn_id).rjust(6, '0')}.label")
                self.flist.append((fname_pcds, fname_labels, seq_id, f"{str(fn_id).rjust(6, '0')}.bin"))
        
        print('Validation Samples: ', len(self.flist))
    
    def form_batch(self, pcds_total):
        #quantize
        pcds_xyzi = pcds_total[:, :4]
        pcds_sem_label = pcds_total[:, 4]
        pcds_ins_label = pcds_total[:, 5]
        pcds_offset = utils.gene_point_offset(pcds_total, center_type=self.config.center_type)
        pcds_coord = utils.Quantize(pcds_xyzi,
                                    range_x=self.Voxel.range_x,
                                    range_y=self.Voxel.range_y,
                                    range_z=self.Voxel.range_z,
                                    size=self.Voxel.bev_shape)

        pcds_sphere_coord = utils.SphereQuantize(pcds_xyzi,
                                            phi_range=(-180.0, 180.0),
                                            theta_range=self.Voxel.RV_theta,
                                            size=self.Voxel.rv_shape)

        #convert numpy matrix to pytorch tensor
        pcds_xyzi = make_point_feat(pcds_xyzi, pcds_coord, pcds_sphere_coord, self.Voxel)
        pcds_xyzi = torch.FloatTensor(pcds_xyzi.astype(np.float32))
        pcds_xyzi = pcds_xyzi.transpose(1, 0).contiguous()

        pcds_coord = torch.FloatTensor(pcds_coord.astype(np.float32))
        pcds_sphere_coord = torch.FloatTensor(pcds_sphere_coord.astype(np.float32))

        pcds_sem_label = torch.LongTensor(pcds_sem_label.astype(np.int64))
        pcds_ins_label = torch.LongTensor(pcds_ins_label.astype(np.int64))
        pcds_offset = torch.FloatTensor(pcds_offset.astype(np.float32))
        return pcds_xyzi.unsqueeze(-1), pcds_coord.unsqueeze(-1), pcds_sphere_coord.unsqueeze(-1), pcds_sem_label.unsqueeze(-1), pcds_ins_label.unsqueeze(-1), pcds_offset
    
    def __getitem__(self, index):
        fname_pcds, fname_labels, seq_id, fn = self.flist[index]

        #load point clouds and label file
        pcds = np.fromfile(fname_pcds, dtype=np.float32)
        pcds = pcds.reshape((-1, 4))

        pcds_label = np.fromfile(fname_labels, dtype=np.uint32)
        pcds_label = pcds_label.reshape((-1))

        sem_label = pcds_label & 0xFFFF
        inst_label = pcds_label >> 16

        pcds_label_use = utils.relabel(sem_label, self.task_cfg['learning_map'])
        pcds_ins_label = utils.gene_ins_label(pcds_label_use, inst_label)
        
        pano_label = (inst_label << 16) + pcds_label_use
        # merge pcds and labels
        pcds_total = np.concatenate((pcds, pcds_label_use[:, np.newaxis], pcds_ins_label[:, np.newaxis]), axis=1)

        # data aug
        pcds_xyzi_list = []
        pcds_coord_list = []
        pcds_sphere_coord_list = []
        pcds_sem_label_list = []
        pcds_ins_label_list = []
        pcds_offset_list = []
        for x_sign in [1, -1]:
            for y_sign in [1, -1]:
                pcds_tmp = pcds_total.copy()
                pcds_tmp[:, 0] *= x_sign
                pcds_tmp[:, 1] *= y_sign
                pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset = self.form_batch(pcds_tmp)

                pcds_xyzi_list.append(pcds_xyzi)
                pcds_coord_list.append(pcds_coord)
                pcds_sphere_coord_list.append(pcds_sphere_coord)
                pcds_sem_label_list.append(pcds_sem_label)
                pcds_ins_label_list.append(pcds_ins_label)
                pcds_offset_list.append(pcds_offset)
        
        pcds_xyzi = torch.stack(pcds_xyzi_list, dim=0)
        pcds_coord = torch.stack(pcds_coord_list, dim=0)
        pcds_sphere_coord = torch.stack(pcds_sphere_coord_list, dim=0)
        pcds_sem_label = torch.stack(pcds_sem_label_list, dim=0)
        pcds_ins_label = torch.stack(pcds_ins_label_list, dim=0)
        pcds_offset = torch.stack(pcds_offset_list, dim=0)
        pano_label = torch.LongTensor(pano_label.astype(np.int64))
        return pcds_xyzi, pcds_coord, pcds_sphere_coord, pcds_sem_label, pcds_ins_label, pcds_offset, pano_label, seq_id, fn
    
    def __len__(self):
        return len(self.flist)