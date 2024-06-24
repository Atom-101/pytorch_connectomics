import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
import os
import glob

from connectomics.data.dataset import VolumeDataset

class MadiDataset(VolumeDataset):
    def __init__(self,
                cfg,
                augmentor,
                mode,
                no_weight,
                df,
                sample_volume_size,
                sample_label_size,
                sample_stride,
                target_opt,
                weight_opt,
                data_mean,
                data_std,
                erosion_rates,
                dilation_rates,
                label_idx,
            ):
        super().__init__(
            [],
            augmentor = augmentor,
            sample_volume_size=sample_volume_size,
            sample_label_size=sample_label_size,
            sample_stride=sample_stride,
            target_opt=target_opt,
            weight_opt=weight_opt,
            data_mean=data_mean,
            data_std=data_std,
            data_match_act=None,
            erosion_rates=erosion_rates,
            dilation_rates=dilation_rates,
            do_relabel=False,
        )

        self.cfg = cfg
        self.label_idx = label_idx
        self.mode=mode
        self.no_weight = no_weight

        self.data_path = '/n/home11/abanerjee/UW_Madison/data/train_3d/images'
        # self.files = glob.glob(os.path.join(self.data_path, '/*.npz'))

        self.df = df

        dfs = [y.reset_index(drop=True) for x,y in self.df.groupby(['case', 'day'])]
        self.files = [os.path.join(self.data_path, f'case{df.iloc[0, 2]}_day{df.iloc[0, 3]}.npz') for df in dfs]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file = np.load(file)
        # import pdb; pdb.set_trace()
        
        x, y = file['x'].astype(np.float32)/16383, file['y'][..., self.label_idx].astype(np.uint8)

        # filtering
        if x.shape[0] == 144:
            x,y = x[24:138],y[24:138]
        elif x.shape[0] == 80:
            x,y = x[2:77],y[2:77]

        if self.mode == 'train':
            if x.shape[0] > 96:
                idx_start = np.random.choice(np.arange(0, x.shape[0] - 96 + 1))
                idx_end = idx_start + 96
            else:
                idx_start = 0
                idx_end = x.shape[0]

            x, y = x[idx_start:idx_end], y[idx_start:idx_end]

            xsplit = np.split(x, x.shape[0], axis=0)
            t = self.augmentor(image=xsplit[0].squeeze(0),
                masks=[l.squeeze(0) for l in np.split(y, y.shape[0], axis=0)],
                **{f'image_{n:03d}':i.squeeze(0) for i,n in zip(xsplit[1:], range(len(xsplit)-1))})
            
            x = np.stack([t[k] for k in sorted([k for k in t.keys() if 'image' in k])])
            y = np.stack(t['masks'])

            x = F.interpolate(torch.from_numpy(x)[None], (320, 320), mode='bilinear', align_corners=False)[0].numpy()
            y = F.interpolate(torch.from_numpy(y)[None], (320, 320), mode='nearest')[0].numpy()

            # data = {'image':x, 'label':y}
        else:
            # data = {'image':x, 'label':y}
            pass

        sample = self._process_targets((None, x, y, None))
        if self.no_weight:
            sample = list(sample)
            sample[-1] = [np.ones_like(x) for _ in range(len(sample[-1]))]
            sample = tuple(sample)
        # import pdb; pdb.set_trace()

        return sample

class MadiDatasetSeg(VolumeDataset):
    def __init__(self,
                cfg,
                augmentor,
                mode,
                no_weight,
                df,
                sample_volume_size,
                sample_label_size,
                sample_stride,
                target_opt,
                weight_opt,
                data_mean,
                data_std,
                erosion_rates,
                dilation_rates,
                label_idx,
            ):
        super().__init__(
            [],
            augmentor = augmentor,
            sample_volume_size=sample_volume_size,
            sample_label_size=sample_label_size,
            sample_stride=sample_stride,
            target_opt=target_opt,
            weight_opt=weight_opt,
            data_mean=data_mean,
            data_std=data_std,
            data_match_act=None,
            erosion_rates=erosion_rates,
            dilation_rates=dilation_rates,
            do_relabel=False,
        )

        self.cfg = cfg
        self.label_idx = label_idx
        self.mode=mode
        self.no_weight = no_weight

        self.data_path = '/n/home11/abanerjee/UW_Madison/data/train_3d/images'
        # self.files = glob.glob(os.path.join(self.data_path, '/*.npz'))

        self.df = df

        dfs = [y.reset_index(drop=True) for x,y in self.df.groupby(['case', 'day'])]
        self.files = [os.path.join(self.data_path, f'case{df.iloc[0, 2]}_day{df.iloc[0, 3]}.npz') for df in dfs]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        file = np.load(file)
        # import pdb; pdb.set_trace()
        
        x, y = file['x'].astype(np.float32)/16383, file['y'].astype(np.uint8)

        # filtering
        if x.shape[0] == 144:
            x,y = x[24:138],y[24:138]
        elif x.shape[0] == 80:
            x,y = x[2:77],y[2:77]

        if self.mode == 'train':
            if x.shape[0] > 96:
                idx_start = np.random.choice(np.arange(0, x.shape[0] - 96 + 1))
                idx_end = idx_start + 96
            else:
                idx_start = 0
                idx_end = x.shape[0]

            x, y = x[idx_start:idx_end], y[idx_start:idx_end]

            xsplit = np.split(x, x.shape[0], axis=0)
            ysplit = [i.squeeze(0).squeeze(2) for j in np.split(y, y.shape[0], axis=0) for i in np.split(j, 3, axis=-1)]
            t = self.augmentor(image=xsplit[0].squeeze(0),
                masks=ysplit,
                **{f'image_{n:03d}':i.squeeze(0) for i,n in zip(xsplit[1:], range(len(xsplit)-1))})
            
            x = np.stack([t[k] for k in sorted([k for k in t.keys() if 'image' in k])]) # z, h, w
            yt = [np.stack(t['masks'][i:i+3]) for i in range(0, len(t['masks']), 3)]  # [3, h, w]
            y = np.stack(yt).transpose(1,0,2,3)  # z, 3, h, w -> 3,z,h,w

            x = F.interpolate(torch.from_numpy(x)[None], (320, 320), mode='bilinear', align_corners=False).numpy()
            y = F.interpolate(torch.from_numpy(y), (320, 320), mode='nearest').numpy()

            # data = {'image':x, 'label':y}
        else:
            # data = {'image':x, 'label':y}
            x = x[None]
            y = y.transpose(3,0,1,2)

        return 4 * (x - 0.0448), y  # dataloader will give 5d outs



