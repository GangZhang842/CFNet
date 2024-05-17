import os
import torch
import numpy as np

import pdb


class BaseRecorder(object):
    def __init__(self, save_dir: str, version: str = 'version', save_topk_model: int = 1, mode: str = 'max', **kwargs):
        if version is None:
            version = 'version'
        kwargs.update(locals())
        kwargs.pop('kwargs')
        kwargs.pop('self')
        if '__class__' in kwargs:
            kwargs.pop('__class__')
        for key in kwargs:
            setattr(self, key, kwargs[key])
        
        self.process_group = torch.distributed.group.WORLD
        self.log_dir = None
        self.checkpoint_dir = None
        if (self.process_group is None) or (torch.distributed.get_rank(self.process_group) == 0):
            assert self.mode in ['max', 'min']
            i = 0
            log_dir = os.path.join(save_dir, "{0}_{1}".format(version, i))
            while os.path.exists(log_dir):
                i = i + 1
                log_dir = os.path.join(save_dir, "{0}_{1}".format(version, i))
            self.log_dir = log_dir
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            os.system("mkdir -p {}".format(self.log_dir))
            os.system("mkdir -p {}".format(self.checkpoint_dir))
        
        if (self.process_group is not None) and (torch.distributed.get_world_size(self.process_group) >= 2):
            self.sync_all_process()
            i = 0
            log_dir = os.path.join(save_dir, "{0}_{1}".format(version, i))
            while os.path.exists(log_dir):
                i = i + 1
                log_dir = os.path.join(save_dir, "{0}_{1}".format(version, i))
            
            self.log_dir = os.path.join(save_dir, "{0}_{1}".format(version, i-1))
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        
        self.metric_list = []
        self.rebuild()
    
    def rebuild(self):
        pass
    
    def sync_all_process(self):
        torch.distributed.barrier(self.process_group)
    
    def _record(self, *args, **kwargs):
        raise NotImplementedError
    
    def record(self, *args, **kwargs):
        if (self.process_group is None) or (torch.distributed.get_rank(self.process_group) == 0):
            self._record(*args, **kwargs)
    
    def _record_str(self, *args, **kwargs):
        raise NotImplementedError
    
    def record_str(self, *args, **kwargs):
        if (self.process_group is None) or (torch.distributed.get_rank(self.process_group) == 0):
            self._record_str(*args, **kwargs)
    
    def _record_checkpoint(self, metric: float, epoch_idx: int):
        meta_info = [epoch_idx, round(metric, 8)]
        meta_info.append(os.path.join(self.checkpoint_dir, "epoch={0}-metric={1}.cpkt".format(*tuple(meta_info))))
        self.metric_list.append(meta_info)

        fname_checkpoint = None
        if len(self.metric_list) > self.save_topk_model:
            epoch_array = np.array([x[0] for x in self.metric_list], dtype=np.int32)
            metric_array = np.array([x[1] for x in self.metric_list], dtype=np.float32)

            reorder_index = np.argsort(metric_array, axis=0, kind='stable')
            if self.mode == 'max':
                reorder_index = reorder_index[::-1]
            
            metric_array = metric_array[reorder_index]
            epoch_array = epoch_array[reorder_index]

            if epoch_idx in epoch_array[:self.save_topk_model].tolist():
                fname_checkpoint = self.metric_list[-1][2]
            
            # delete model
            for i in reorder_index[self.save_topk_model:].tolist():
                if os.path.exists(self.metric_list[i][2]):
                    os.remove(self.metric_list[i][2])
        else:
            fname_checkpoint = self.metric_list[-1][2]
        
        return fname_checkpoint
    
    def record_checkpoint(self, *args, **kwargs):
        if (self.process_group is None) or (torch.distributed.get_rank(self.process_group) == 0):
            return self._record_checkpoint(*args, **kwargs)