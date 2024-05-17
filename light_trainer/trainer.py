import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import optimizer, lr_scheduler
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import collections
import logging
import random
import yaml
import sys

from tqdm import tqdm
import os

from .recorder import base_recorder

import pdb


_LOGGER = logging.getLogger()


@torch.no_grad()
def reduce_tensor(inp, group=None):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = torch.distributed.get_world_size(group)
    if world_size < 2:
        return inp
    
    if isinstance(inp, torch.Tensor):
        reduce_inp = inp.cpu()
    else:
        reduce_inp = inp
    
    object_list = [None for i in range(world_size)]
    torch.distributed.all_gather_object(object_list, reduce_inp, group=group)
    return sum(object_list) / world_size


def init_env(seed=0, enable_dist=True):
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if enable_dist:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        process_group = torch.distributed.group.WORLD
        if process_group is not None:
            local_rank = int(os.getenv("LOCAL_RANK"))
            torch.cuda.set_device(local_rank)


def dic_to_human_readable_str(title_name, param_dic):
    assert isinstance(param_dic, dict)
    string = "{}:\n".format(title_name)
    for key in param_dic:
        string += "{0}:\t{1}\n".format(key, param_dic[key])
    return string


def attach_input_device(batch):
    if isinstance(batch, (tuple, list)):
        result = []
        for x in batch:
            if isinstance(x, torch.Tensor):
                result.append(x.cuda())
            else:
                result.append(x)
        return result
    elif isinstance(batch, dict):
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
        return batch
    else:
        raise NotImplementedError


def convert_tuple_to_list(params):
    if isinstance(params, dict):
        for key in params:
            params[key] = convert_tuple_to_list(params[key])
        return params
    elif isinstance(params, (list, tuple)):
        params = list(params)
        for i in range(len(params)):
            params[i] = convert_tuple_to_list(params[i])
        return params
    else:
        return params


class ADDistTrainer:
    def __init__(self,
        recorder: base_recorder.BaseRecorder = None,
        max_epochs: int = 100,
        precision: str = "fp32",
        log_every_n_steps: int = 1,
        check_val_every_n_epoch: int = 1,
        clip_grad_norm: float = None,
        clip_grad_norm_type: int = 2,
        detect_anomaly: bool = False,
        sync_batchnorm: bool = False,
        **kwargs):
        # capture arguments to provide to context
        assert precision in ["fp32", "fp16", "bf16"], "'precision' must be fp32, fp16, or bf16, but got {}".format(precision)
        kwargs.update(locals())
        kwargs.pop('kwargs')
        kwargs.pop('self')
        if '__class__' in kwargs:
            kwargs.pop('__class__')
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.enable_fp16 = (precision == "fp16") or (precision == "bf16")
        self.float16_type = None
        if precision == "fp16":
            self.float16_type = torch.float16
        elif precision == "bf16":
            self.float16_type = torch.bfloat16
        
        self.build_env()
        self.build_training_prop()

        # save settings
        self.log_str(dic_to_human_readable_str("Project Settings", kwargs))
    
    def build_env(self):
        # set env
        self.process_group = torch.distributed.group.WORLD
        if self.process_group is not None:
            self.local_rank = int(os.getenv("LOCAL_RANK"))
            self.world_size = torch.distributed.get_world_size(self.process_group)
            self.global_rank = torch.distributed.get_rank(self.process_group)
            torch.cuda.set_device(self.local_rank)
    
    def build_training_prop(self):
        # set training properties
        self.model: nn.Module = None
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        self.optimizer: Union[optimizer.Optimizer, Dict[str, optimizer.Optimizer]] = None
        self.scheduler: Union[lr_scheduler._LRScheduler, Dict[str, lr_scheduler._LRScheduler]] = None
        self.global_step_idx: int = 0
        self.step_idx: int = 0
        self.epoch_idx: int = 0
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enable_fp16)
    
    def set_attrs(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
    
    def save_hyperparameters(self, param_dic: Dict[str, Any], fn_yaml: str):
        if (self.process_group is not None) and (self.global_rank == 0) and (self.recorder is not None):
            assert isinstance(param_dic, dict)
            fname_yaml = os.path.join(self.recorder.log_dir, fn_yaml)

            param_dic_convert = convert_tuple_to_list(param_dic)
            with open(fname_yaml, 'w') as f:
                f.write(yaml.dump(param_dic_convert, allow_unicode=True))
    
    def log(self, enable_sync_dist: bool = True, ignore_log_step: bool = False, **kwargs):
        if (((self.step_idx % self.log_every_n_steps) == 0) or ignore_log_step) and (self.recorder is not None):
            if enable_sync_dist:
                kwargs_sync = collections.OrderedDict()
                for key in kwargs:
                    kwargs_sync[key] = reduce_tensor(kwargs[key], self.process_group)
                
                self.recorder.record(kwargs_sync, self.epoch_idx, self.step_idx, self.global_step_idx)
            else:
                self.recorder.record(kwargs, self.epoch_idx, self.step_idx, self.global_step_idx)
    
    def log_str(self, string: str):
        if self.recorder is not None:
            self.recorder.record_str(string)
    
    def configure_optimizers(self):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    # training
    def training_epoch_start(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def training_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def training_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    # validation
    def validation_epoch_start(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def validation_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def validation_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    # test
    def test_epoch_start(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def test_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def test_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    # predict
    def predict_epoch_start(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def predict_step(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def predict_epoch_end(self, *args, **kwargs):
        print("{} should be re-implemented if used!".format(sys._getframe().f_code.co_name))
    
    def load_from_checkpoint(self,
                            fname_ckpt: str,
                            is_load_model: bool = True,
                            is_load_optimizer: bool = True,
                            is_load_scheduler: bool = True,
                            strict:bool = True):
        saved_dic = torch.load(fname_ckpt, map_location='cpu')
        if is_load_model:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(saved_dic['model_dic'], strict=strict)
            else:
                self.model.load_state_dict(saved_dic['model_dic'], strict=strict)
        
        if is_load_optimizer:
            self.optimizer.load_state_dict(saved_dic['optimizer'])
        
        if is_load_scheduler:
            self.scheduler.load_state_dict(saved_dic['scheduler'])
    
    def save_checkpoint(self, fname_ckpt: str):
        model_dic = None
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_dic = self.model.module.state_dict()
        else:
            model_dic = self.model.state_dict()
        
        saved_dic = {
            'model_dic': model_dic,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step_idx': self.global_step_idx,
            'step_idx': self.step_idx,
            'epoch_idx': self.epoch_idx
        }
        torch.save(saved_dic, fname_ckpt)
    
    def resume_from_checkpoint(self, fname_ckpt: str):
        saved_dic = torch.load(fname_ckpt, map_location='cpu')
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.module.load_state_dict(saved_dic['model_dic'], strict=True)
        else:
            self.model.load_state_dict(saved_dic['model_dic'], strict=True)
        
        self.optimizer.load_state_dict(saved_dic['optimizer'])
        self.scheduler.load_state_dict(saved_dic['scheduler'])
        self.global_step_idx = saved_dic['global_step_idx']
        self.step_idx = saved_dic['step_idx']
        self.epoch_idx = saved_dic['epoch_idx']
    
    def sync_all_process(self):
        torch.distributed.barrier(self.process_group)
    
    def fit(self,
            model: nn.Module,
            train_dataloader: Optional[DataLoader],
            val_dataloader: Optional[DataLoader] = None,
            ckpt_path: str = None,
            find_unused_parameters = False):
        # distributed model and dataloader
        assert isinstance(model, nn.Module), "'model' must belong to nn.Module, but got {}".format(type(model))
        self.model = model
        if self.sync_batchnorm:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        self.model = torch.nn.parallel.DistributedDataParallel(self.model.cuda(),
                                                    device_ids=[self.local_rank],
                                                    output_device=self.local_rank,
                                                    find_unused_parameters=find_unused_parameters)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        # configure optimizer and scheduler
        self.configure_optimizers()

        # resume from checkpoint
        if ckpt_path is not None:
            self.resume_from_checkpoint(ckpt_path)
        
        # logger model, optimizer, scheduler
        if self.global_rank == 0:
            _LOGGER.info(model)
            _LOGGER.info(self.optimizer)
            _LOGGER.info(self.scheduler)
        
        # judge optimizer and scheduler
        if not isinstance(self.optimizer, optimizer.Optimizer):
            assert isinstance(self.optimizer, dict) and \
            all([isinstance(self.optimizer[key], optimizer.Optimizer) for key in self.optimizer]), \
            "self.optimizer must be torch.optim.optimizer.Optimizer or dict[str, torch.optim.optimizer.Optimizer]"
        
        if not isinstance(self.scheduler, lr_scheduler._LRScheduler):
            assert isinstance(self.scheduler, dict) and \
            all([isinstance(self.scheduler[key], lr_scheduler._LRScheduler) for key in self.scheduler]), \
            "self.scheduler must be torch.optim.lr_scheduler._LRScheduler or dict[str, torch.optim.lr_scheduler._LRScheduler]"
        
        # start fitting
        while(self.epoch_idx < self.max_epochs):
            # training
            self.model.train()
            if hasattr(self.train_dataloader, "sampler"):
                self.train_dataloader.sampler.set_epoch(self.epoch_idx)
            
            self.step_idx = 0

            # training epoch start
            self.sync_all_process()
            self.training_epoch_start()
            self.sync_all_process()

            with torch.autograd.set_detect_anomaly(self.detect_anomaly):
                self.log_str("Training Epoch: {}".format(self.epoch_idx))

                loop = enumerate(self.train_dataloader)
                if self.global_rank == 0:
                    loop = tqdm(loop, desc="Training Epoch: {}".format(self.epoch_idx), total=len(self.train_dataloader), ascii=True)
                
                for batch_idx, batch in loop:
                    with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16, dtype=self.float16_type):
                        loss = self.training_step(batch_idx, attach_input_device(batch))
                    
                    if loss is not None:
                        assert (isinstance(loss, torch.Tensor) and loss.numel() == 1)
                        # automatic backward
                        self.optimizer.zero_grad()
                        self.scaler.scale(loss).backward()
                        if self.clip_grad_norm is not None:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm, norm_type=self.clip_grad_norm_type)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                    
                    self.global_step_idx += 1
                    self.step_idx += 1
            
            # training epoch end
            self.sync_all_process()
            self.training_epoch_end()
            self.sync_all_process()

            # evaluate
            if self.val_dataloader is not None:
                if self.epoch_idx % self.check_val_every_n_epoch == 0:
                    # validation epoch start
                    self.sync_all_process()
                    self.validation_epoch_start()
                    self.sync_all_process()

                    self.model.eval()
                    self.log_str("Evaluating Epoch: {}".format(self.epoch_idx))
                    loop = enumerate(self.val_dataloader)
                    if self.global_rank == 0:
                        loop = tqdm(loop, desc="Evaluating Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
                    
                    for batch_idx, batch in loop:
                        with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16, dtype=self.float16_type):
                            self.validation_step(batch_idx, attach_input_device(batch))
                    
                    # validation epoch end
                    self.sync_all_process()
                    metric = self.validation_epoch_end()
                    self.sync_all_process()
                    assert isinstance(metric, (int, float))
                    fname_ckpt = self.recorder.record_checkpoint(metric, self.epoch_idx)
                    if (self.global_rank == 0) and (fname_ckpt is not None):
                        self.save_checkpoint(fname_ckpt)
            else:
                if self.global_rank == 0:
                    fname_ckpt = os.path.join(self.recorder.checkpoint_dir, "epoch={0}.cpkt".format(self.epoch_idx))
                    self.save_checkpoint(fname_ckpt)
            
            self.sync_all_process()
            self.epoch_idx += 1
    
    @torch.no_grad()
    def validate(self, model: nn.Module, dataloader: Optional[DataLoader] = None, ckpt_path: str = None):
        # distributed model and dataloader
        assert isinstance(model, nn.Module), "'model' must belong to nn.Module, but got {}".format(type(model))
        self.model = model
        if dataloader is not None:
            self.val_dataloader = dataloader
        
        # resume from checkpoint
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path, is_load_optimizer=False, is_load_scheduler=False)
        
        # validation epoch start
        self.sync_all_process()
        self.validation_epoch_start()
        self.sync_all_process()

        # evaluate
        self.model.cuda()
        self.model.eval()
        loop = enumerate(self.val_dataloader)
        if self.global_rank == 0:
            loop = tqdm(loop, desc="Evaluating Epoch: {}".format(self.epoch_idx), total=len(self.val_dataloader), ascii=True)
        
        for batch_idx, batch in loop:
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16, dtype=self.float16_type):
                self.validation_step(batch_idx, attach_input_device(batch))
        
        # validation epoch end
        self.sync_all_process()
        metric = self.validation_epoch_end()
        self.sync_all_process()
    
    @torch.no_grad()
    def test(self, model: nn.Module, dataloader: Optional[DataLoader], ckpt_path: str = None):
        # distributed model and dataloader
        assert isinstance(model, nn.Module), "'model' must belong to nn.Module, but got {}".format(type(model))
        self.model = model
        # resume from checkpoint
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path, is_load_optimizer=False, is_load_scheduler=False)
        
        # test epoch end
        self.sync_all_process()
        self.test_epoch_start()
        self.sync_all_process()

        # test
        self.model.cuda()
        self.model.eval()
        loop = enumerate(dataloader)
        if self.global_rank == 0:
            loop = tqdm(loop, desc="Testing Epoch: {}".format(self.epoch_idx), total=len(dataloader), ascii=True)
        
        for batch_idx, batch in loop:
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16, dtype=self.float16_type):
                self.test_step(batch_idx, attach_input_device(batch))
        
        # test epoch end
        self.sync_all_process()
        self.test_epoch_end()
        self.sync_all_process()
    
    @torch.no_grad()
    def predict(self, model: nn.Module, dataloader: Optional[DataLoader], ckpt_path: str = None):
        # distributed model and dataloader
        assert isinstance(model, nn.Module), "'model' must belong to nn.Module, but got {}".format(type(model))
        self.model = model
        # resume from checkpoint
        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path, is_load_optimizer=False, is_load_scheduler=False)
        
        # predict epoch end
        self.sync_all_process()
        self.predict_epoch_start()
        self.sync_all_process()

        # predict
        self.model.cuda()
        self.model.eval()
        loop = enumerate(dataloader)
        if self.global_rank == 0:
            loop = tqdm(loop, desc="Predicting Epoch: {}".format(self.epoch_idx), total=len(dataloader), ascii=True)
        
        for batch_idx, batch in loop:
            with torch.cuda.amp.autocast(enabled=self.enable_fp16, cache_enabled=self.enable_fp16, dtype=self.float16_type):
                self.predict_step(batch_idx, attach_input_device(batch))
        
        # predict epoch end
        self.sync_all_process()
        self.predict_epoch_end()
        self.sync_all_process()