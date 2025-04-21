import os
import json
import torch
import argparse
import torch.nn as nn
from datasets.cifar10 import get_dataloader
from models.quant import (
    replace_conv_linear,
    set_aux_opt,
    AuxOpt
)
from types import SimpleNamespace
from models.rconvmixers import ConvMixer
from models.param_mng import (
    parameter_manager,
    scale_wrapper, 
    read_raw_data_1t1r_8,
    read_raw_data_2t2r_15,
)


@torch.no_grad()
def eval_epoch():
    net.eval()
    total_loss = 0
    total_samples = 0
    acc_samples = 0
    total_batches = len(test_loader)
    for i, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        B = inputs.size(0)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * B

        total_samples += B
        pred_class = torch.argmax(outputs, dim=-1)
        acc_samples += (pred_class == targets).sum().item()
        print("[INFO] Evaluating... Batch: {}/{}, Avg loss: {:.4f}, Avg acc: {:.4f}"
              .format(i+1, total_batches, 
                      total_loss/total_samples, acc_samples/total_samples), end="\r")
    print()
    loss_avg = total_loss / total_samples
    acc_avg = acc_samples / total_samples
    return loss_avg, acc_avg


parser = argparse.ArgumentParser()
parser.add_argument("ckpt_dir", type=str)
opt = parser.parse_args()

ckpt_folder: str = opt.ckpt_dir
aux_json = os.path.join(ckpt_folder, "aux.json")
launch_json = os.path.join(ckpt_folder, "launch.json")
ckpt_path = os.path.join(ckpt_folder, "ckpt_latest.pt")

with open(launch_json, "r") as fp:
    launch_opt = json.load(fp)
aux_opt = AuxOpt.from_file(aux_json)


criterion = nn.CrossEntropyLoss()
opt = SimpleNamespace(**launch_opt)
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
if opt.diff:
    parameter_manager.switch_to_fn(scale_wrapper(read_raw_data_2t2r_15))
else:
    parameter_manager.switch_to_fn(scale_wrapper(read_raw_data_1t1r_8))

net = ConvMixer(opt.hdim)
if opt.quant:
    net = replace_conv_linear(net)
    parameter_manager.to(device)
net.to(device)
net.load_state_dict(torch.load(ckpt_path, map_location=device))
set_aux_opt(net, aux_opt)


train_loader, test_loader = get_dataloader(
    data_root="./data/CIFAR10",
    batch_size=opt.batch_size, 
    num_workers=opt.workers
)
print("-"*80)
test_loss, test_acc = eval_epoch()

