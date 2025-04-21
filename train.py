import os
import time
import json
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from datasets.cifar10 import get_dataloader
from torch.utils.tensorboard import SummaryWriter
from models.quant import (
    replace_conv_linear,
    set_aux_opt,
    AuxOpt
)
from models.rconvmixers import ConvMixer
from models.param_mng import (
    parameter_manager,
    scale_wrapper, 
    read_raw_data_1t1r_8,
    read_raw_data_2t2r_15,
)


def parse_args():
    parser = argparse.ArgumentParser()
    # config network:
    parser.add_argument("--hdim", type=int, default=128)
    parser.add_argument("--quant", action="store_true", default=False)
    parser.add_argument("--diff", action="store_true", default=False)
    # config dataset
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    # config training
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=5e-4, help="weight decay")
    # checkpoint saving
    parser.add_argument("--save", type=str, default="")
    return parser.parse_args()


def save_launch_opt(opt, path: str):
    save_dir = os.path.dirname(path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    with open(path, "w") as fp:
        json.dump(vars(opt), fp, indent=4, ensure_ascii=False)


def train_epoch():
    net.train()
    total_loss = 0
    total_samples = 0
    acc_samples = 0
    total_batches = len(train_loader)
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)

        B = inputs.size(0)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * B
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_samples += B
        pred_class = torch.argmax(outputs, dim=-1)
        acc_samples += (pred_class == targets).sum().item()

        current_lr = scheduler.get_last_lr()[0] if scheduler is not None else opt.lr
        print(("[INFO] Epoch: {}/{}, Batch: {}/{}, Avg loss: {:.4f}, Avg acc: {:.4f}"
               ", lr: {:.3e}")
              .format(epoch+1, opt.epochs, i+1, total_batches, 
                      total_loss/total_samples, acc_samples/total_samples,
                      current_lr), end="\r")
    print()
    loss_avg = total_loss / total_samples
    acc_avg = acc_samples / total_samples
    return loss_avg, acc_avg


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


def get_save_folder():
    if opt.save:
        ckpt_dir = os.path.join("checkpoints", opt.save)
        log_dir = os.path.join("logs", opt.save)
        print("[INFO] Checkpoints saved to {}".format(os.path.abspath(ckpt_dir)))
        print("[INFO] Logs saved to {}".format(os.path.abspath(log_dir)))
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    else:
        ckpt_dir = None
        log_dir = None
    return ckpt_dir, log_dir


if __name__ == "__main__":
    opt = parse_args()

    criterion = nn.CrossEntropyLoss()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

    if opt.diff:
        parameter_manager.switch_to_fn(scale_wrapper(read_raw_data_2t2r_15))
    else:
        parameter_manager.switch_to_fn(scale_wrapper(read_raw_data_1t1r_8))

    net = ConvMixer(dim=opt.hdim)
    if opt.quant:
        net = replace_conv_linear(net)
        parameter_manager.to(device)
    net.to(device)

    aux_opt = AuxOpt()
    set_aux_opt(net, aux_opt)

    train_loader, test_loader = get_dataloader(
        data_root="./data/CIFAR10",
        batch_size=opt.batch_size, 
        num_workers=opt.workers
    )
    optimizer = optim.AdamW(net.parameters(), lr=opt.lr, weight_decay=opt.wd)

    scheduler = None
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=opt.lr, epochs=opt.epochs, steps_per_epoch=len(train_loader)
    )

    ckpt_dir, log_dir = get_save_folder()
    if opt.save:
        writer = SummaryWriter(log_dir, flush_secs=60)
        aux_opt.to_file(os.path.join(ckpt_dir, "aux.json"))
        save_launch_opt(opt, os.path.join(ckpt_dir, "launch.json"))

    for epoch in range(opt.epochs):
        print("-"*80)
        train_loss, train_acc = train_epoch()
        test_loss, test_acc = eval_epoch()
        if scheduler is not None:
            last_lr = scheduler.get_last_lr()[0]
        else:
            last_lr = opt.lr

        if opt.save:
            ckpt_file = os.path.join(ckpt_dir, "ckpt_latest.pt")
            torch.save(net.state_dict(), ckpt_file)
            writer.add_scalar("lr", last_lr, epoch)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/acc", train_acc, epoch)
            writer.add_scalar("test/loss", test_loss, epoch)
            writer.add_scalar("test/acc", test_acc, epoch)
    time.sleep(5)

