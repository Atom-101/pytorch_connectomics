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

from dataset import MadiDataset
import albumentations as A

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

        self.optimizer = build_optimizer(self.cfg, self.model)
        self.lr_scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.scaler = GradScaler() if cfg.MODEL.MIXED_PRECESION else None
        self.start_iter = self.cfg.MODEL.PRE_MODEL_ITER

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
        self.criterion = Criterion.build_from_cfg(self.cfg, self.device)
        
        self.monitor = build_monitor(self.cfg)
        self.monitor.load_info(self.cfg, self.model)

        self.total_iter_nums = self.cfg.SOLVER.ITERATION_TOTAL - self.start_iter
        self.total_time = 0

        self.dataset, self.dataloader = None, None

        sample_label_size = cfg.MODEL.OUTPUT_SIZE
        topt, wopt = ['0'], [['0']]
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        sample_stride = (1, 1, 1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = 50
        self.is_main_process = True
        self.best_val_loss = float('inf')

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

        self.train_dataset = MadiDataset(
            self.cfg, self.augmentor, 'train', no_weight=False,
            df=pd.read_csv('/n/home11/abanerjee/UW_Madison/data/train_set.csv', delimiter='; '), 
            **shared_kwargs)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, pin_memory=True, shuffle=True,
                                                            batch_size=1, collate_fn=collate_fn_train, 
                                                            num_workers=4)

        self.val_dataset = MadiDataset(
            self.cfg, None, 'val', no_weight=False,
            df=pd.read_csv('/n/home11/abanerjee/UW_Madison/data/val_set.csv', delimiter='; '),
            **shared_kwargs)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, pin_memory=True, shuffle=False,
                                                          batch_size=1, collate_fn=collate_fn_train,
                                                          num_workers=4)

        self.output_dir = cfg.DATASET.OUTPUT_PATH
        # import pdb; pdb.set_trace()

    
    def train(self):
        r"""Training function of the trainer class.
        """
        self.model.train()

        for i in range(50):
            iter_total = self.start_iter + i * len(self.train_loader)
            self.start_time = time.perf_counter()
            self.optimizer.zero_grad()

            for sample_idx, sample in enumerate(tqdm(self.train_loader)):
                # load data
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l
                self.data_time = time.perf_counter() - self.start_time

                # prediction
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.model(volume)
                    loss, losses_vis = self.criterion(pred, target, weight)

                self._train_misc(loss, pred, volume, target, weight,
                                iter_total + sample_idx, losses_vis)
            self.validate(i)
            self.save_checkpoint(-1)  # always keep last checkpoint

    def validate(self, epoch):
        r"""Validation function of the trainer class.
        """
        if not hasattr(self, 'val_loader'):
            return

        self.model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for i, sample in enumerate(tqdm(self.val_loader)):
                volume = sample.out_input
                target, weight = sample.out_target_l, sample.out_weight_l

                # prediction
                volume = volume.to(self.device, non_blocking=True)
                with autocast(enabled=self.cfg.MODEL.MIXED_PRECESION):
                    pred = self.model(volume)
                    loss, _ = self.criterion(pred, target, weight)
                    val_loss += loss.data
        
        print(f'Val loss at end of epoch-{epoch}: {val_loss}')

        # import pdb; pdb.set_trace()
        if hasattr(self, 'monitor'):
            self.monitor.logger.log_tb.add_scalar(
                'Validation_Loss', val_loss, epoch)
            weight[0][0] = weight[0][0][:, :, 44:]
            self.monitor.visualize(volume[:,:,44:, : ,:], [target[0][:,:,44:, : ,:]], pred[:,:,44:, : ,:],
                                   weight, epoch, suffix='Val')
            print('logged images')

        if not hasattr(self, 'best_val_loss'):
            self.best_val_loss = val_loss

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch, is_best=True)
            print('Best loss saved')


        # Release some GPU memory and ensure same GPU usage in the consecutive iterations according to
        # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770
        del pred, loss, val_loss

        # model.train() only called at the beginning of Trainer.train().
        self.model.train()

    # def is_main_process(self):
    #     return True


def main():
    args = get_args()
    label_idx = int(args.opts[1])
    args.opts = []
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
    trainer = MadiTrainer(cfg, checkpoint, label_idx)
    trainer.train()
    # trainer.validate(0)


if __name__ == "__main__":
    seed_everything(42)
    main()