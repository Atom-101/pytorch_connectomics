import torch
from torch.cuda.amp import autocast, GradScaler

import os
import time
import math
import GPUtil
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from typing import *

from connectomics.engine import Trainer
from connectomics.data.augmentation import build_train_augmentor
from model import build_madi_model

from connectomics.model.loss import Criterion
from connectomics.utils.monitor import build_monitor
from connectomics.engine.solver import *
from connectomics.engine import TrainerBase
from connectomics.utils.system import get_args, init_devices
from connectomics.config import load_cfg, save_all_cfg

from connectomics.data.dataset.collate import collate_fn_train

from dataset import MadiDataset, MadiDatasetSeg
import albumentations as A

import torch.nn as nn
import torch.nn.functional as F

out_size = 320


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class MadiTrainer(Trainer):
    def __init__(self,
                 cfg,
                 checkpoint = None,
                 label_idx = 0,
                 mode = 'train'):
        
        # TrainerBase.__init__(self, cfg, torch.device('cuda:0'))
        self.cfg = cfg
        self.device = torch.device('cuda:0')
        self.model = build_madi_model(self.cfg, self.device, None)
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        # self.optimizer = build_optimizer(self.cfg, self.model)
        param_groups = [
            {'params': self.model.backbone.parameters(), 'weight_decay': 1e-3},
            {'params': self.model.middle_convs.parameters(), 'weight_decay': 1e-3},
            {'params': self.model.smooth.parameters(), 'weight_decay': 1e-3},
            {'params': self.model.convs.parameters(), 'weight_decay': 1e-3},
            {'params': self.model.post_up_conv.parameters(), 'weight_decay': 1e-3},
            {'params': self.model.res_block.parameters(), 'weight_decay': 5e-4},
            {'params': self.model.conv_out.parameters(), 'weight_decay': 5e-4},
        ]
        self.optimizer = torch.optim.AdamW(param_groups, lr=0)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                            self.optimizer, max_lr=[5e-4, 5e-4, 8e-4, 8e-4, 8e-4, 1e-3, 1e-3],
                                            total_steps=cfg.SOLVER.ITERATION_TOTAL, last_epoch=-1, pct_start=0.15)
        self.scaler = GradScaler()

        # self.augmentor = build_train_augmentor(self.cfg)
        self.augmentor = A.Compose([
            A.ColorJitter(brightness=(1, 1.2), contrast=0.3, saturation=0.3, hue=0.1, p=0.8),
            # A.ToGray(p=1.0),
            A.Affine(scale=None, rotate=(-30, 30), shear=(-5, 5), 
                    translate_percent={'x':(-0.1, 0.1), 'y':(-0.1, 0.1)}, p=0.3),
            A.RandomResizedCrop(out_size, out_size, scale=(0.75, 1.0), ratio=(0.75, 1.33), p=0.3),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur(p=0.5),
            A.OneOf([
                    A.GridDistortion(p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
                ], p=0.25),
            A.CoarseDropout(max_holes=8, max_height=out_size//20, max_width=out_size//20,
                                min_holes=3, fill_value=0, mask_fill_value=0, p=0.25),
            # A.OneOf([
            #         A.CoarseDropout(max_holes=8, max_height=16, max_width=16,
            #                      min_holes=5, fill_value=0, mask_fill_value=0, p=1.0),
            #         A.MaskDropout(max_objects=3, p=1.0)
            #     ], p=0.3),
            ],
            additional_targets={f'image_{n:03d}': 'image' for n in range(100)} 
        )

        sample_label_size = cfg.MODEL.OUTPUT_SIZE
        topt, wopt = ['0'], [['0']]
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        sample_stride = (1, 1, 1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = 50
        self.is_main_process = True
        self.best_iou = 0

        shared_kwargs = {
            "sample_volume_size": sample_volume_size,
            "sample_label_size": sample_label_size,
            "sample_stride": sample_stride,
            "target_opt": topt,
            "weight_opt": wopt,
            "data_mean": 0.0448,
            "data_std": 0.25,
            "erosion_rates": cfg.MODEL.LABEL_EROSION,
            "dilation_rates": cfg.MODEL.LABEL_DILATION,
            "label_idx": label_idx  # [0,1,2]
        }

        self.train_dataset = MadiDatasetSeg(
            self.cfg, self.augmentor, 'train', no_weight=True,
            df=pd.read_csv('/n/home11/abanerjee/UW_Madison/data/train_set.csv', delimiter='; '), 
            **shared_kwargs)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, pin_memory=True, shuffle=True,
                                                            batch_size=1,
                                                            num_workers=4)

        self.val_dataset = MadiDatasetSeg(
            self.cfg, None, 'val', no_weight=True,
            df=pd.read_csv('/n/home11/abanerjee/UW_Madison/data/val_set.csv', delimiter='; '),
            **shared_kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, pin_memory=True, shuffle=False,
                                                          batch_size=1,
                                                          num_workers=4)

        self.output_dir = cfg.DATASET.OUTPUT_PATH
        # import pdb; pdb.set_trace()

    
    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for epoch in range(150):
            self.optimizer.zero_grad()

            for sample_idx, sample in enumerate(tqdm(self.train_loader)):
                # load data
                x, y = sample
                x= x.cuda()
                y = y.cuda().float()

                with autocast(enabled=True):
                    pred = self.model(x)

                    floss = focal_loss(pred, y)
                    dloss = dice_loss(pred.sigmoid(), y)

                    loss = floss + 0.5 * dloss

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            print(f'Epoch-{epoch+1} | Loss: {loss:.5f}, LR: {self.lr_scheduler.get_last_lr()[0]:.5f}, Focal Loss: {floss:.5f}, Dice Loss: {dloss:.5f}')
            self.validate(epoch)


    def validate(self, epoch):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        iou = 0
        iou_per_class = {1:0, 2:0, 3:0}
        count_per_class = {1:0, 2:0, 3:0}

        with torch.no_grad():
            val_loss = 0.0
            for i, sample in enumerate(tqdm(self.val_loader)):
                x, y = sample
                x = x.cuda()
                
                with autocast(enabled=True):
                    pred = self.model(x)
                    
                pred = pred.view(pred.shape[0], 3, -1).float().cpu().sigmoid() > 0.5
                y = y.view(y.shape[0], 3, -1)

                for class_idx in range(3):
                    p = pred[:, class_idx]  # bs, dxhxw
                    t = y[:, class_idx] > 0

                    inter = 2 * (p & t).long().sum(-1)
                    union = p.long().sum(-1) + t.long().sum(-1)
                    count_per_class[class_idx+1] += (t.long().sum(-1) > 0).long().sum()
                    iou_per_class[class_idx+1] += (inter/(1e-3 + union)).sum()
            
            for k in iou_per_class.keys():
                iou_per_class[k] = iou_per_class[k]/count_per_class[k]
                iou += iou_per_class[k]
            iou = iou/3  
        
        print(f'mIoU at epoch-{epoch+1}: {iou}')
        print(f'Classwise IoU at epoch-{epoch+1}: {iou_per_class}')


        if iou >= self.best_iou:
            self.best_iou = iou
            self.save_checkpoint(epoch, is_best=True)
            print('Best IoU saved')


        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del pred, iou_per_class

        # model.train() only called at the beginning of Trainer.train().
        self.model.train()


def focal_loss(x, y):
    wts = torch.tensor([42.6921, 38.2337, 37.3982]).reshape(1, -1, 1, 1, 1).cuda()
    loss = F.binary_cross_entropy_with_logits(x, y, reduction = 'none')

    loss = (wts.shape[1] * (loss * wts).mean(dim=[1,2,3,4]))/wts.sum()

    return loss.mean()

def dice_loss(x, y):
    tp = x * y
    fp = x * (1 - y)
    fn = (1 - x) * y

    tp = tp.sum(dim=(2,3,4))
    fp = fp.sum(dim=(2,3,4))
    fn = fn.sum(dim=(2,3,4))

    tversky = (tp + 50.0) / (tp + 0.3*fp + 0.7*fn + 50.0)
    tversky = tversky.mean()

    return 1-tversky



def main():
    args = get_args()
    cfg = load_cfg(args)

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)['state_dict']
    else:
        checkpoint = None

    print("PyTorch: ", torch.__version__)
    print(cfg)

    if not os.path.exists(cfg.DATASET.OUTPUT_PATH):
        print('Output directory: ', cfg.DATASET.OUTPUT_PATH)
        os.makedirs(cfg.DATASET.OUTPUT_PATH)
        save_all_cfg(cfg, cfg.DATASET.OUTPUT_PATH)

    # label_idx = int(sys.argv[1])
    trainer = MadiTrainer(cfg, checkpoint)
    trainer.train()
    # trainer.validate(0)


if __name__ == "__main__":
    seed_everything(42)
    main()