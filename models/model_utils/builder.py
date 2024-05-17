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


def get_scheduler(optimizer, pSch, per_epoch_num_iters):
    num_epoch = pSch.max_epochs
    if pSch.type == 'OneCycle':
        base_lr = optimizer.state_dict()['param_groups'][0]['lr']
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=base_lr,
                                            epochs=num_epoch, steps_per_epoch=per_epoch_num_iters,
                                            pct_start=pSch.pct_start, anneal_strategy='cos',
                                            div_factor=25, final_div_factor=base_lr / pSch.final_lr)
        return scheduler
    elif pSch.type == 'step':
        scheduler = lr_scheduler.LambdaLR(
            optimizer, lr_lambda=partial(
                schedule_with_warmup,
                num_epoch=num_epoch,
                per_epoch_num_iters=per_epoch_num_iters,
                pct_start=pSch.pct_start,
                step=pSch.step,
                decay_factor=pSch.decay_factor
            ))
        return scheduler
    else:
        raise NotImplementedError(pSch.type)


def get_optimizer(pOpt, model):
    if pOpt.type in ['adam', 'adamw']:
        optimizer = AdamW(params=model.parameters(),
                        lr=pOpt.base_lr,
                        weight_decay=pOpt.wd)
        return optimizer
    elif pOpt.type == 'sgd':
        optimizer = SGD(params=model.parameters(),
                        lr=pOpt.base_lr,
                        momentum=pOpt.momentum,
                        weight_decay=pOpt.wd,
                        nesterov=pOpt.nesterov)
        return optimizer
    else:
        raise NotImplementedError(pOpt.type)