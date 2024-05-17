import datasets
import models

import inspect
import pdb


def class2dic(config):
    assert inspect.isclass(config)
    config_dic = dict(config.__dict__)
    del_key_list = []
    for key in config_dic:
        if key.startswith('__') and key.endswith('__'):
            del_key_list.append(key)
    
    for key in del_key_list:
        config_dic.pop(key)
    return config_dic


def class2dic_iterative(config):
    assert inspect.isclass(config)
    config_dic = dict(config.__dict__)
    del_key_list = []
    for key in config_dic:
        if key.startswith('__') and key.endswith('__'):
            del_key_list.append(key)
    
    for key in del_key_list:
        config_dic.pop(key)
    
    for key in config_dic:
        if inspect.isclass(config_dic[key]):
            config_dic[key] = class2dic_iterative(config_dic[key])
    return config_dic


def get_module(user_cfg=None, *args, **kwargs):
    if user_cfg != None:
        if inspect.isclass(user_cfg):
            user_cfg = class2dic(user_cfg)
        elif isinstance(user_cfg, str):
            user_cfg = dict(type=user_cfg)
        else:
            assert isinstance(user_cfg, dict)
        
        for key in user_cfg:
            kwargs[key] = user_cfg[key]
    
    assert 'type' in kwargs
    module_code = eval(kwargs['type'])
    del kwargs['type']
    
    sig = inspect.signature(module_code.__init__)
    input_param_name_list = []
    is_save_extra_params = False
    for param in sig.parameters.values():
        if param.name == 'self':
            continue
        elif param.kind == param.VAR_POSITIONAL:
            raise TypeError("{0} '__init__' should not have positional variables.".format(module_code))
        elif param.kind == param.VAR_KEYWORD:
            is_save_extra_params = True
        else:
            input_param_name_list.append(param.name)
    
    new_kwargs = {}
    for i, value in enumerate(args):
        new_kwargs[input_param_name_list[i]] = value
    
    for key in kwargs:
        if (key in input_param_name_list) or is_save_extra_params:
            new_kwargs[key] = kwargs[key]
    
    result_module = module_code(**new_kwargs)
    return result_module