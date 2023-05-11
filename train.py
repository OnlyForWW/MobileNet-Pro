"""
PyTroch-CNN训练脚本
code by WJY
本代码包含多种Tricks，具体请见代码
代码已经过Linux平台测试，可以直接运行
在运行代码前请提前安装：PyTorch、timm、torchmetrics、tqdm以及apex
具体运行环境：
- os: ubuntu20.04
- Python: 3.8
- PyTroch: 1.11.0
- GPU: RTX A5000 24G
- CPU: Intel(R) Xeon(R) Platinum 8350C CPU
- Memory: 42G
"""

import argparse
import torch
import timm
import timm.data
import timm.loss
import timm.optim
import timm.utils
import torchmetrics
import os
import time
from torch.utils.data import DataLoader
from MobileNetPro import *
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from apex import amp

def create_train_dataload(train_transforms, train_path, batch_size):
    train_dataset = timm.data.dataset.ImageDataset(
        train_path,
        transform=train_transforms
    )

    train_dataload = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    return train_dataload

def main(data_path):

    img_size = (224, 224)

    lr = 5e-3

    smoothing = 0.1

    mixup = 0.2
    cutmix = 1.0

    batch_size = 128

    bce_target_thresh = 0.2

    num_epoch = 150

    data_path = Path(data_path)

    train_path = data_path / "train"
    val_path = data_path / "val"

    num_classes = len(list(train_path.iterdir()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mixup_fn = timm.data.Mixup(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        label_smoothing=smoothing,
        num_classes=num_classes
    )

    model = MobileNetPro_100(num_classes=7)
    model = model.to(device=device)

    data_mean = [0.485, 0.456, 0.406]
    data_std = [0.229, 0.224, 0.225]

    train_transforms_mid = timm.data.create_transform(
        input_size=img_size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m8-n4-mstd0.5-inc1"
    )

    val_transforms = timm.data.create_transform(
        input_size=img_size,
        mean=data_mean,
        std=data_std
    )

    val_dataset = timm.data.dataset.ImageDataset(
        val_path,
        transform=val_transforms
    )

    val_dataload = DataLoader(dataset=val_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    optimizer = timm.optim.create_optimizer_v2(
        model,
        opt="lookahead_AdamW",
        lr=lr,
        weight_decay=0.01
    )

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_epoch,
        cycle_decay=0.5,
        lr_min=1e-6,
        t_in_epochs=True,
        warmup_t=5,
        warmup_lr_init=1e-4,
        cycle_limit=1
    )

    train_loss_fn = timm.loss.BinaryCrossEntropy(
        target_threshold=bce_target_thresh,
        smoothing=smoothing
    )
    train_loss_fn = train_loss_fn.to(device=device)

    val_loss_fn = torch.nn.CrossEntropyLoss()
    val_loss_fn = val_loss_fn.to(device=device)

    ema_model = timm.utils.ModelEmaV2(model=model, decay=0.9, device=device)

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=7)
    metric.to(device=device)

    ema_metric = torchmetrics.Accuracy(task="multiclass", num_classes=7)
    ema_metric.to(device=device)

    model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level="O1")

    train_best_acc = None
    ema_best_acc = None

    start_time = time.time()

    for epoch in range(num_epoch):
        print(f"——————epoch: {epoch + 1} starting.——————")

        train_dataload = create_train_dataload(train_transforms=train_transforms_mid,
                                               train_path=train_path,
                                               batch_size=batch_size)

        num_steps_per = len(train_dataload)
        num_updates = epoch * num_steps_per

        model.train()
        train_epoch_loss = 0.0
        for batch in tqdm(train_dataload, desc="train"):
            imgs, targets = batch
            imgs = imgs.to(device=device)
            targets = targets.to(device=device)

            mixup_imgs, mixup_targets = mixup_fn(imgs, targets)

            preds = model(imgs)

            train_loss = train_loss_fn(preds, mixup_targets)

            optimizer.zero_grad()

            with amp.scale_loss(train_loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            train_epoch_loss += scaled_loss.item()
            optimizer.step()
            scheduler.step_update(num_updates=num_updates)
        scheduler.step(epoch+1)
        ema_model.update(model)
        if hasattr(optimizer, "sync_lookahead"):
            optimizer.sync_lookahead()

        model.eval()
        ema_model.eval()
        val_epoch_loss = 0.0
        for batch in tqdm(val_dataload, desc="val"):
            imgs, targets = batch
            imgs = imgs.to(device=device)
            targets = targets.to(device=device)

            preds = model(imgs)
            ema_preds = ema_model.module(imgs)
            batch_val_loss = val_loss_fn(preds, targets)
            val_epoch_loss += batch_val_loss.item()

            acc = metric(preds, targets)

            ema_acc = ema_metric(ema_preds, targets)

        acc = metric.compute()
        ema_acc = ema_metric.compute()

        print(f"epoch{epoch + 1}, acc={acc}, ema_acc={ema_acc}.")
        print(f"——————epoch: {epoch + 1} ended.——————")

        if train_best_acc == None or train_best_acc < acc:
            train_best_acc = acc
            torch.save(model.state_dict(), './best.pth')

        if ema_best_acc == None or ema_best_acc < ema_acc:
            ema_best_acc = ema_acc
            torch.save(ema_model.state_dict(), './best_ema.pth')

        metric.reset()
        ema_metric.reset()

    total_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))
    with open("results.txt", 'a') as f:
        f.write(f"train_best_acc: {train_best_acc}\nema_best_acc: {ema_best_acc}\ntime: {total_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script with some tricks.")
    parser.add_argument("--data_dir", required=True, help="The dataset folder on disk.")
    args = parser.parse_args()

    main(args.data_dir)

    os.system("/usr/bin/shutdown")