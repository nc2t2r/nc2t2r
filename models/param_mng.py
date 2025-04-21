import os
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Callable


def read_raw_data_1t1r_8():
    excel_file = "read_model/1T1R_8states.xlsx"

    df = pd.read_excel(excel_file)
    data = df.to_numpy()[:, ::-1]

    medians = torch.as_tensor(np.median(data, axis=0), dtype=torch.float64)
    means = torch.as_tensor(np.mean(data, axis=0), dtype=torch.float64)
    stds = torch.as_tensor(np.std(data, axis=0), dtype=torch.float64)

    return medians, means, stds


def read_raw_data_2t2r_15():
    excel_file = "read_model/2T2R_16_states.xlsx"

    df = pd.read_excel(excel_file)

    available_columns: List[str] = []
    for c in df.columns:
        if "=" in c:
            available_columns.append(c)
    
    data = {}
    for c in available_columns:
        level = int(c.split("=")[0])
        if level not in data:
            data[level] = []
        data[level].append(df[c].to_numpy())

    data_means = []
    data_medians = []
    data_stds = []

    for level, samples in data.items():
        if len(samples) == 0:
            data_medians.append(np.median(samples[0]))
            data_means.append(np.mean(samples[0]))
            data_stds.append(np.std(samples[0]))
        else:
            medians = [np.median(s) for s in samples]
            means = [np.mean(s) for s in samples]
            stds = [np.std(s) for s in samples]
            data_medians.append(np.mean(medians))
            data_means.append(np.mean(means))
            data_stds.append(np.mean(stds))
    
    data_medians = np.asarray(data_medians)
    data_means = np.asarray(data_means)
    data_stds = np.asanyarray(data_stds)

    sort_indices = np.argsort(data_means)
    data_medians = data_medians[sort_indices]
    data_means = data_means[sort_indices]
    data_stds = data_stds[sort_indices]

    medians = torch.from_numpy(data_medians).to(torch.float64)
    means = torch.from_numpy(data_means).to(torch.float64)
    stds = torch.from_numpy(data_stds).to(torch.float64)

    return medians, means, stds



class _ParameterManager(object):
    def __init__(self):
        self.fn2param: Dict[str, List[torch.Tensor]] = {}
        self.fn = None
    
    def register_fn(self, fn: Callable):
        print(fn())
        self.fn2param[fn.__name__] = list(fn())
    
    def switch_to_fn(self, fn: Callable):
        if fn.__name__ not in self.fn2param:
            self.register_fn(fn)
        self.fn = fn
    
    def to(self, *args, **kwagrs):
        assert self.fn is not None
        params = self.fn2param[self.fn.__name__]
        for i in range(len(params)):
            params[i] = params[i].to(*args, **kwagrs)
    
    @property
    def states(self):
        assert self.fn is not None
        return self.fn2param[self.fn.__name__]
    
    @property
    def states_medians(self):
        assert self.fn is not None
        return self.fn2param[self.fn.__name__][0]
    
    @property
    def states_means(self):
        assert self.fn is not None
        return self.fn2param[self.fn.__name__][1]
    
    @property
    def states_stds(self):
        assert self.fn is not None
        return self.fn2param[self.fn.__name__][2]


def scale_wrapper(fn: Callable):
    def fn_wrapped():
        states_medians, states_mean, states_std = fn()
        scale = 1.0 / states_mean.abs().max()
        outs = states_medians * scale, states_mean * scale, states_std * scale
        outs = tuple(o.float() for o in outs)
        return outs
    fn_wrapped.__name__ = fn.__name__
    return fn_wrapped


parameter_manager = _ParameterManager()


if __name__ == "__main__":
    parameter_manager.switch_to_fn(read_raw_data_2t2r_15)
    medians, means, stds = parameter_manager.states
    print(medians)
    print(means)
    print(stds)

