import os
import torch
import numpy as np
from .base_recorder import BaseRecorder

from typing import Any, Dict, Iterable, List, Optional, Union

import logging
from datetime import datetime
from pytz import utc, timezone


def config_logger(path):
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Asia/Shanghai")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    
    logging.basicConfig()
    logging.getLogger().handlers.pop()

    fmt = '%(asctime)s %(message)s'
    date_fmt = '%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    formatter.converter = custom_time

    logging.getLogger().setLevel(logging.INFO)

    log_file_save_name = path
    file_handler = logging.FileHandler(filename=log_file_save_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


class TXTRecorder(BaseRecorder):
    def rebuild(self):
        self.fname_record = os.path.join(self.log_dir, 'log.txt')
        config_logger(self.fname_record)
        self.logger = logging.getLogger()
    
    def _record(self, record_dict: Dict[str, Any], epoch_idx: int, step_idx: int, global_step_idx: int):
        string = 'Epoch: {0}; Iteration: {1}'.format(epoch_idx, step_idx)
        for key in record_dict:
            value = record_dict[key]
            if isinstance(value, (torch.Tensor, np.ndarray)):
                string += "; {0}: {1}".format(key, value.mean().item())
            else:
                string += "; {0}: {1}".format(key, value)
        
        self._record_str(string)
    
    def _record_str(self, string: str):
        self.logger.info(string)