import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import pdb

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets

from models import *

import pytorch_lib

import tqdm
import importlib
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel, pOpt = config.get_config()
    
    prefix = pGen.name
    save_path = os.path.join("experiments", prefix)
    model_prefix = os.path.join(save_path, "checkpoint")

    #define dataloader
    test_dataset = eval('datasets.{}.DataloadTest'.format(pDataset.Val.data_src))(pDataset.Val)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    test_loader = iter(test_loader)

    #define model
    model = eval(pModel.prefix).AttNet(pModel)
    model.eval()
    model.cuda()

    pv_nms = pytorch_lib.PointVoteNMS(model.point_nms_dic)

    # load pretrain model
    pretrain_model = os.path.join(model_prefix, '{}-model.pth'.format(args.test_epoch))
    model.load_state_dict(torch.load(pretrain_model, map_location='cpu'))
    print('Load model: {}'.format(pretrain_model))

    pcds_xyzi, pcds_coord, pcds_sphere_coord, seq_id, fn = test_loader.next()

    pcds_xyzi = pcds_xyzi[0, [0]].contiguous().cuda()
    pcds_coord = pcds_coord[0, [0]].contiguous().cuda()
    pcds_sphere_coord = pcds_sphere_coord[0, [0]].contiguous().cuda()
    pdb.set_trace()

    time_cost_model = []
    time_cost_post = []
    with torch.no_grad():
        for i in range(1000):
            start = time.time()

            torch.cuda.synchronize()
            pred_sem, pred_offset, pred_hmap = model.infer(pcds_xyzi, pcds_coord, pcds_sphere_coord)
            torch.cuda.synchronize()

            start1 = time.time()
            
            torch.cuda.synchronize()
            pred_sem = F.softmax(pred_sem, dim=1)[0, :, :, 0].T
            pv_nms(pcds_xyzi[0, :3, :, 0].T, pred_sem, pred_offset[0], pred_hmap[0, :, 0])
            torch.cuda.synchronize()

            end = time.time()

            time_cost_model.append(start1 - start)
            time_cost_post.append(end - start1)
    
    print('Time Model: ', np.array(time_cost_model[20:]).mean(), 'Time Post: ', np.array(time_cost_post[20:]).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_epoch', type=int, default=30)
    
    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)