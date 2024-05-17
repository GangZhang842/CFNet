import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import datasets
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.config_parser import get_module
from light_trainer import trainer, recorder

import argparse
import importlib


import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = False

import pdb


def save_dataset(dataloader):
    import pickle as pkl
    process_group = torch.distributed.group.WORLD
    global_rank = torch.distributed.get_rank(process_group)
    data_list = []
    for data in dataloader:
        seq_id = data[-2]
        fn = data[-1]
        key = f"{seq_id}/{fn}"
        data_list.append(key)
    
    print(global_rank, len(set(data_list)), len(data_list))


def save_dataloader(dataloader, fpath):
    import pickle as pkl
    process_group = torch.distributed.group.WORLD
    global_rank = torch.distributed.get_rank(process_group)
    data_list = []
    if not os.path.exists(fpath):
        os.system("mkdir -p {}".format(fpath))
    
    fname_pkl = os.path.join(fpath, f"rank_{global_rank}.pkl")
    for data in dataloader:
        seq_id_list = data[-2]
        fn_list = data[-1]
        for i in range(len(seq_id_list)):
            key = f"{seq_id_list[i]}/{fn_list[i]}"
            data_list.append(key)
    
    print(global_rank, len(set(data_list)), len(data_list))
    with open(fname_pkl, "wb") as f:
        pkl.dump(data_list, f)


def main(args, config):
    # parsing cfg
    pGen, pDataset, pModel = config.get_config()

    prefix = pGen.name
    trainer.init_env(seed=5120)

    # define dataloader
    train_dataset = get_module(type=pDataset.Train.type, config=pDataset.Train)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset,
                            batch_size=pGen.batch_size_per_gpu,
                            shuffle=(train_sampler is None),
                            num_workers=pDataset.Train.num_workers,
                            sampler=train_sampler,
                        )
    
    val_dataset = get_module(type=pDataset.Val.type, config=pDataset.Val)
    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=(val_sampler is None),
                            num_workers=pDataset.Val.num_workers,
                            sampler=val_sampler,
                        )
    
    # define recorder, model, and trainer
    txt_recorder = recorder.txt_recorder.TXTRecorder(
        save_dir=os.path.join("./experiments", pGen.name),
        version=args.version,
        save_topk_model=5,
        mode='max')
    
    model = get_module(type=pModel.type, pModel=pModel)
    model_trainer = eval(pModel.runner_type)(
        recorder=txt_recorder,
        max_epochs=pModel.scheduler.max_epochs,
        precision=args.precision,
        log_every_n_steps=args.log_frequency,
        sync_batchnorm=True,
        pModel=pModel,
        per_epoch_num_iters=len(train_loader)
    )
    if (args.pretrain_model != None) and (os.path.exists(args.pretrain_model)):
        print('Load pretrain model: {}'.format(args.pretrain_model))
        model.load_state_dict(torch.load(args.pretrain_model, map_location='cpu')['model_dic'], strict=False)
    
    model_trainer.fit(model, train_dataloader=train_loader, val_dataloader=val_loader, ckpt_path=args.resume_ckpt, find_unused_parameters=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='lidar panoptic segmentation')
    parser.add_argument('--config', help='config file path', type=str)
    parser.add_argument('--log_frequency', help='number of devices', type=int, default=100)
    parser.add_argument('--precision', help='precision of the float number', type=str, default="fp32")

    parser.add_argument('--pretrain_model', help='pretrain model', type=str, default=None)
    parser.add_argument('--resume_ckpt', help='resume checkpoint', type=str, default=None)
    parser.add_argument('--version', help='version name of the experiment', type=str, default="version")

    args = parser.parse_args()
    config = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    main(args, config)