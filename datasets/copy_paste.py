import numpy as np
import random
import os

from scipy.spatial.transform import Rotation as R
from scipy.spatial import Delaunay

import pdb


def in_range(v, r):
    return (v >= r[0]) * (v < r[1])


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def compute_box_3d(center, size, yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    
    # 3d bounding box dimensions
    l = size[0]
    w = size[1]
    h = size[2]
    
    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    
    corners_3d[0, :] = corners_3d[0, :] + center[0]
    corners_3d[1, :] = corners_3d[1, :] + center[1]
    corners_3d[2, :] = corners_3d[2, :] + center[2]
    return corners_3d.T


def rotate_along_z(pcds, theta):
    rotateMatrix = R.from_euler('z', theta, degrees=True).as_matrix()[:2, :2].T
    pcds[:, :2] = pcds[:, :2].dot(rotateMatrix)
    return pcds


def random_f(r):
    return r[0] + (r[1] - r[0]) * random.random()


class CutPaste:
    def __init__(self, object_dir, paste_max_obj_num, road_idx=[9]):
        self.object_dir = object_dir
        self.sub_dirs = ('car', 'other-vehicle', 'truck', 'motorcyclist', 'motorcycle', 'person', 'bicycle', 'bicyclist')
        '''
        other-vehicle: 7014
        bicycle: 4063
        motorcyclist: 530
        bicyclist: 1350
        motorcycle: 2774
        truck: 2514
        person: 6764
        '''
        self.sub_dirs_dic = {}
        for fp in self.sub_dirs:
            fpath = os.path.join(self.object_dir, fp)
            fname_list = [os.path.join(fpath, x) for x in os.listdir(fpath) if (x.endswith('.npz')) and (x.split('_')[0] != '08')]
            print('Load {0}: {1}'.format(fp, len(fname_list)))
            fname_list = sorted(fname_list)
            self.sub_dirs_dic[fp] = fname_list
        
        self.paste_max_obj_num = paste_max_obj_num
        self.road_idx = road_idx
    
    def get_random_rotate_along_z_obj(self, pcds_obj, bbox_corners, theta):
        pcds_obj_result = rotate_along_z(pcds_obj, theta)
        bbox_corners_result = rotate_along_z(bbox_corners, theta)
        return pcds_obj_result, bbox_corners_result

    def get_fov(self, pcds_obj):
        x, y, z = pcds_obj[:, 0], pcds_obj[:, 1], pcds_obj[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        u_fov = (u.min(), u.max())
        phi_fov = (phi.min(), phi.max())
        theta_fov = (theta.min(), theta.max())
        return u_fov, phi_fov, theta_fov

    def occlusion_process(self, pcds, phi_fov, theta_fov):
        x, y, z = pcds[:, 0], pcds[:, 1], pcds[:, 2]
        d = np.sqrt(x ** 2 + y ** 2 + z ** 2) + 1e-12
        u = np.sqrt(x ** 2 + y ** 2) + 1e-12

        phi = np.arctan2(x, y)
        theta = np.arcsin(z / d)

        fov_mask = in_range(phi, phi_fov) * in_range(theta, theta_fov)
        return fov_mask

    def paste_single_obj(self, pcds, pcds_road, pcds_label, pcds_ins_label):
        '''
        Input:
            pcds, (N, 4), 4 -> x, y, z, intensity
            pcds_road, (M, 4)
            pcds_label, (N,)
            pcds_ins_label, (N,)
        Output:
            pcds, (N1, 4)
            pcds_label, (N1,)
        '''
        # pcds (N, 4), 4 contains x, y, z, intensity
        # pcds_label(N)
        cate = random.choice(self.sub_dirs)
        fname_npz = random.choice(self.sub_dirs_dic[cate])
        npkl = np.load(fname_npz)

        pcds_obj = npkl['pcds']
        cate_id = int(npkl['cate_id'])
        semantic_cate = str(npkl['cate'])

        bbox_center = npkl['center']
        bbox_size = npkl['size'] * 1.05
        bbox_yaw = npkl['yaw']
        bbox_corners = compute_box_3d(bbox_center, bbox_size, bbox_yaw)

        if(len(pcds_obj) < 10):
            return pcds, pcds_label, pcds_ins_label
        
        theta_list = np.arange(0, 360, 18).tolist()
        np.random.shuffle(theta_list)
        for theta in theta_list:
            # global rotate object
            pcds_obj_aug, bbox_corners_aug = self.get_random_rotate_along_z_obj(pcds_obj, bbox_corners, theta)

            # get local road height
            valid_road_mask = in_hull(pcds_road[:, :2], bbox_corners_aug[:4, :2])
            pcds_local_road = pcds_road[valid_road_mask]
            if pcds_local_road.shape[0] > 5:
                road_mean_height = float(pcds_local_road[:, 2].mean())
                pcds_obj_aug[:, 2] = pcds_obj_aug[:, 2] + (road_mean_height - pcds_obj_aug[:, 2].min())
            else:
                # object is not on road
                continue
            
            # get object fov
            u_fov, phi_fov, theta_fov = self.get_fov(pcds_obj_aug)
            if (abs(u_fov[1] - u_fov[0]) < 8) and (abs(phi_fov[1] - phi_fov[0]) < 1) and (abs(theta_fov[1] - theta_fov[0]) < 1):
                # get valid fov
                fov_mask = self.occlusion_process(pcds, phi_fov, theta_fov)
                in_fov_obj_mask = in_range(pcds_label[fov_mask], (1, 9))
                if in_fov_obj_mask.sum() < 3:
                    assert pcds.shape[0] == pcds_label.shape[0]
                    assert pcds.shape[0] == pcds_ins_label.shape[0]

                    # add object back
                    pcds_filter = pcds[~fov_mask]
                    pcds_label_filter = pcds_label[~fov_mask]
                    pcds_ins_label_filter = pcds_ins_label[~fov_mask]

                    pcds = np.concatenate((pcds_filter, pcds_obj_aug), axis=0)

                    pcds_addobj_label = np.full((pcds_obj_aug.shape[0],), fill_value=cate_id, dtype=pcds_label.dtype)
                    pcds_label = np.concatenate((pcds_label_filter, pcds_addobj_label), axis=0)

                    pcds_addobj_ins_label = np.full((pcds_obj_aug.shape[0],), fill_value=pcds_ins_label_filter.max()+1, dtype=pcds_ins_label_filter.dtype)
                    pcds_ins_label = np.concatenate((pcds_ins_label_filter, pcds_addobj_ins_label), axis=0)
                    break
                else:
                    # invalid heading
                    continue
            else:
                break
        
        return pcds, pcds_label, pcds_ins_label
    
    def __call__(self, pcds, pcds_label, pcds_ins_label):
        paste_obj_num = random.randint(0, self.paste_max_obj_num)
        if paste_obj_num == 0:
            return pcds, pcds_label, pcds_ins_label
        else:
            pcds_road = [pcds[pcds_label == i] for i in self.road_idx]
            pcds_road = np.concatenate(pcds_road, axis=0)

            pcds_new = pcds.copy()
            pcds_label_new = pcds_label.copy()
            pcds_ins_label_new = pcds_ins_label.copy()
            for i in range(paste_obj_num):
                pcds_new, pcds_label_new, pcds_ins_label_new = self.paste_single_obj(pcds_new, pcds_road, pcds_label_new, pcds_ins_label_new)
            return pcds_new, pcds_label_new, pcds_ins_label_new