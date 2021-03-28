# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# src/utils/load_checkpoint.py


import os

import torch


def check_model_state_dict(model, checkpoint):
    return set(model.state_dict().keys()) <= set(checkpoint['state_dict'].keys())


def load_checkpoint(model, optimizer, filename, metric=False, ema=False):
    start_step = 0
    if ema:
        checkpoint = torch.load(filename)
        has_extra_keys = set(model.state_dict().keys()) < set(checkpoint['state_dict'].keys())
        model.load_state_dict(checkpoint['state_dict'], strict=not has_extra_keys)
        return model
    else:
        checkpoint = torch.load(filename)
        seed = checkpoint['seed']
        run_name = checkpoint['run_name']
        start_step = checkpoint['step']
        has_extra_keys = set(model.state_dict().keys()) < set(checkpoint['state_dict'].keys())
        model.load_state_dict(checkpoint['state_dict'], strict=not has_extra_keys)
        if has_extra_keys:
            checkpoint['optimizer']['param_groups'] = optimizer.state_dict()['param_groups']
        optimizer.load_state_dict(checkpoint['optimizer'])
        ada_p = checkpoint['ada_p']
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        if metric:
            best_step = checkpoint['best_step']
            best_fid = checkpoint['best_fid']
            best_fid_checkpoint_path = checkpoint['best_fid_checkpoint_path']
            return model, optimizer, seed, run_name, start_step, ada_p, best_step, best_fid, best_fid_checkpoint_path
    return model, optimizer, seed, run_name, start_step, ada_p
