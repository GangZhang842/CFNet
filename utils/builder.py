import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import collections

from torch.optim import SGD, AdamW, lr_scheduler

from functools import partial
import math

import pdb


def schedule_with_warmup(k, num_epoch, per_epoch_num_iters, pct_start, step, decay_factor):
    warmup_iters = int(num_epoch * per_epoch_num_iters * pct_start)
    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        epoch = k // per_epoch_num_iters
        step_idx = (epoch // step)
        return math.pow(decay_factor, step_idx)


def get_scheduler(optimizer, pOpt, per_epoch_num_iters):
    num_epoch = pOpt.schedule.end_epoch - pOpt.schedule.begin_epoch
    if pOpt.schedule.type == 'OneCycle':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=pOpt.schedule.base_lr,
                                            epochs=num_epoch, steps_per_epoch=per_epoch_num_iters,
                                            pct_start=pOpt.schedule.pct_start, anneal_strategy='cos',
                                            div_factor=25, final_div_factor=pOpt.schedule.base_lr / pOpt.schedule.final_lr)
        return scheduler
    elif pOpt.schedule.type == 'step':
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                schedule_with_warmup,
                num_epoch=num_epoch,
                per_epoch_num_iters=per_epoch_num_iters,
                pct_start=pOpt.schedule.pct_start,
                step=pOpt.schedule.step,
                decay_factor=pOpt.schedule.decay_factor
            ))
        return scheduler
    else:
        raise NotImplementedError(pOpt.schedule.type)


def get_optimizer(pOpt, model):
    if pOpt.optimizer.type in ['adam', 'adamw']:
        optimizer = AdamW(params=model.parameters(),
                        lr=pOpt.optimizer.base_lr,
                        weight_decay=pOpt.optimizer.wd)
        return optimizer
    elif pOpt.optimizer.type == 'sgd':
        optimizer = SGD(params=model.parameters(),
                        lr=pOpt.optimizer.base_lr,
                        momentum=pOpt.optimizer.momentum,
                        weight_decay=pOpt.optimizer.wd,
                        nesterov=pOpt.optimizer.nesterov)
        return optimizer
    else:
        raise NotImplementedError(pOpt.optimizer.type)


class TSEnsemble(object):
    def __init__(self, model, weight_dic, alpha=0.95, device=torch.device('cpu')):
        assert isinstance(model, nn.Module)
        assert (alpha >= 0) and (alpha <= 1)
        self.mean_model = model.to(device)
        self.mean_model.load_state_dict(weight_dic)
        self.alpha = alpha
        self.device = device

    def to(self, *args, **kwargs):
        self.mean_model = self.mean_model.to(*args, **kwargs)

    @torch.no_grad()
    def infer(self, *args, **kwargs):
        self.mean_model.eval()
        args = tuple([x.cuda(self.device) for x in args])
        for key in kwargs:
            kwargs[key] = kwargs[key].cuda(self.device)

        output = self.mean_model.infer_val(*args, **kwargs)
        return output

    def __call__(self, *args, **kwargs):
        return self.infer(*args, **kwargs)

    @torch.no_grad()
    def update(self, model_new):
        assert isinstance(model_new, nn.Module)
        model_new_state_dict = model_new.state_dict()
        old_state_dict = self.mean_model.state_dict()

        new_state_dict = collections.OrderedDict()
        for key in old_state_dict:
            if key in model_new_state_dict:
                new_state_dict[key] = model_new_state_dict[key] * (1 - self.alpha) + old_state_dict[key] * self.alpha
            else:
                raise Exception("{} is not found in model".format(key))
        
        self.mean_model.load_state_dict(new_state_dict)

    def save(self, fname):
        torch.save(self.mean_model.state_dict(), fname)

    def load(self, fname):
        self.mean_model.load_state_dict(torch.load(fname, map_location='cpu'))


def relabel_ignore(pred_cls, thresh=0.6):
    pred_cls_prob = F.softmax(pred_cls, dim=1) #(BS, C, N, 1)
    pred_cls_score, pred_cls_id = pred_cls_prob.max(dim=1) #(BS, N, 1)

    pred_cls_id[pred_cls_score < thresh] = 0
    return pred_cls_id