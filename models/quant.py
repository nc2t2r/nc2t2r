import os
import copy
import json
import torch
from torch.nn import init
from torch import nn, Tensor
from typing import Callable
import torch.nn.functional as F
from .param_mng import parameter_manager


class AuxOpt(object):
    def __init__(
        self,
        dc_train_p = 0.,
        dc_eval_p = 0.,
        dc_when_eval = True,
    
        std_d2d_scale = 0.0, 
        std_scale_eval = 1.0,
        std_when_eval = True,

        decay_eval = 1.0
    ):
        self.dc_train_p = float(dc_train_p)
        self.dc_eval_p = float(dc_eval_p)
        self.dc_when_eval = bool(dc_when_eval)
    
        self.std_d2d_scale = float(std_d2d_scale)
        self.std_scale_eval = float(std_scale_eval)
        self.std_when_eval = bool(std_when_eval)

        self.decay_eval = float(decay_eval)

    def to_file(self, path: str):
        save_dir = os.path.dirname(path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.to_dict(), fp, indent=4, ensure_ascii=False)
    
    @classmethod
    def from_file(cls, path: str):
        with open(path, "r") as fp:
            data = json.load(fp)
        return cls.from_dict(data)
    
    def to_dict(self) -> dict:
        return copy.deepcopy(vars(self))
    
    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
    
    def apply_std(self, mean: Tensor, std: Tensor, training: bool):
        if training:
            s = mean + torch.randn_like(std) * std * (1 + self.std_d2d_scale)
        else:
            if self.std_when_eval:
                s = mean + torch.randn_like(std) * std * self.std_scale_eval * (1 + self.std_d2d_scale)
            else:
                s = mean
        return s
    
    def apply_dc(self, w: Tensor, training: bool):
        if training and self.dc_train_p > 0:
            w = F.dropout(w, p=self.dc_train_p, training=True)
        
        if not training:
            if self.dc_when_eval:
                if self.dc_eval_p > 0:
                    w = F.dropout(w, p=self.dc_eval_p, training=True)
            else:
                w = w  # use training prob
        return w
    
    def apply_decay(self, w: Tensor, training: bool):
        if not training:
            w = w * self.decay_eval
        return w
    
    def op(self, x: Tensor, m: nn.Conv2d, fwd_func: Callable):
        qw_mean, qw_std = QuantImpl.apply(m.weight)
        qw = self.apply_std(qw_mean, qw_std, m.training)
        qw = self.apply_decay(qw, m.training)
        qw = self.apply_dc(qw, m.training)

        if isinstance(m.bias, Tensor):
            qb_mean, qb_std = QuantImpl.apply(m.bias)
            qb = self.apply_std(qb_mean, qb_std, m.training)
            qb = self.apply_dc(qb, m.training)
        else:
            qb = None
        
        out: Tensor = fwd_func(x, qw, qb)
        return out


class QuantImpl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor):
        states_medians, states_mean, states_std = parameter_manager.states
        boundaries = (states_mean[1:] + states_mean[:-1]) * 0.5
        indices = torch.bucketize(w, boundaries)
        qw_mean = states_mean[indices]
        qw_std = states_std[indices]
        return qw_mean, qw_std
    
    @staticmethod
    def backward(ctx, grad_qw_mean, grad_qw_std):
        return grad_qw_mean


class QuantConv(nn.Conv2d):
    def set_aux_opt(self, aux_opt: AuxOpt):
        self.aux_opt = aux_opt

    def reset_parameters(self):
        states_means = parameter_manager.states_means
        min, max = states_means.min(), states_means.max()
        init.uniform_(self.weight, min, max)

    def forward(self, input: Tensor) -> Tensor:
        return self.aux_opt.op(input, self, self._conv_forward)


class QuantLinear(nn.Linear):
    def set_aux_opt(self, aux_opt: AuxOpt):
        self.aux_opt = aux_opt

    def reset_parameters(self):
        states_means = parameter_manager.states_means
        min, max = states_means.min(), states_means.max()
        init.uniform_(self.weight, min, max)
    
    def forward(self, input: Tensor) -> Tensor:
        return self.aux_opt.op(input, self, F.linear)


def replace_submodules(
    root_module: nn.Module, 
    predicate: Callable[[nn.Module], bool], 
    func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    rep_module_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]

    for *parent, k in rep_module_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            try:
                src_module = parent_module[int(k)]
            except Exception as e:
                # src_module = parent_module._modules[k]
                src_module = getattr(parent_module, k)
        else:
            src_module = getattr(parent_module, k)

        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            try:
                parent_module[int(k)] = tgt_module
            except Exception as e:
                # parent_module._modules[k] = tgt_module
                setattr(parent_module, k, tgt_module)
        else:
            setattr(parent_module, k, tgt_module)
    return root_module


def replace_conv_linear(net: nn.Module):
    replace_submodules(
        net,
        predicate=lambda x: isinstance(x, nn.Conv2d),
        func=lambda x: QuantConv(
            in_channels=x.in_channels,
            out_channels=x.out_channels,
            kernel_size=x.kernel_size,
            stride=x.stride, 
            padding=x.padding,
            dilation=x.dilation,
            groups=x.groups,
            bias=x.bias is not None,
            padding_mode=x.padding_mode,
        )
    )
    replace_submodules(
        net,
        predicate=lambda x: isinstance(x, nn.Linear),
        func=lambda x: QuantLinear(
            in_features=x.in_features,
            out_features=x.out_features,
            bias=x.bias is not None
        )
    )
    return net


def set_aux_opt(net: nn.Module, aux_opt: AuxOpt):
    for m in net.modules():
        if isinstance(m, (QuantConv, QuantLinear)):
            m.set_aux_opt(aux_opt)

