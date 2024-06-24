from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from connectomics.model.block import *
from connectomics.model.utils import *
from connectomics.model.utils import IntermediateLayerGetter


def build_madi_model(cfg, device, rank=None):
    kwargs = {
        'block_type': cfg.MODEL.BLOCK_TYPE,
        'in_channel': cfg.MODEL.IN_PLANES,
        'out_channel': cfg.MODEL.OUT_PLANES,
        'filters': cfg.MODEL.FILTERS,
        'ks': cfg.MODEL.KERNEL_SIZES,
        'blocks': cfg.MODEL.BLOCKS,
        'attn': cfg.MODEL.ATTENTION,
        'is_isotropic': cfg.DATASET.IS_ISOTROPIC,
        'isotropy': cfg.MODEL.ISOTROPY,
        'pad_mode': cfg.MODEL.PAD_MODE,
        'act_mode': cfg.MODEL.ACT_MODE,
        'norm_mode': cfg.MODEL.NORM_MODE,
        'pooling': cfg.MODEL.POOLING_LAYER
    }
    
    model = Unet3D(**kwargs)
    return model


class Unet3D(nn.Module):
    def __init__(self,
                 backbone_type: str = 'resnet',
                 block_type: str = 'residual',
                 feature_keys: List[str] = ['feat1', 'feat2', 'feat3', 'feat4', 'feat5'],
                 in_channel: int = 1,
                 out_channel: int = 3,
                 filters: List[int] = [28, 36, 48, 64, 80],
                 ks: List[int] = [3, 3, 5, 3, 3],
                 blocks: List[int] = [2, 2, 2, 2, 2],
                 attn: str = 'squeeze_excitation',
                 is_isotropic: bool = False,
                 isotropy: List[bool] = [False, False, False, True, True],
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 init_mode: str = 'orthogonal',
                 deploy: bool = False,
                 fmap_size=[17, 129, 129],
                 **kwargs):
        super().__init__()
        self.filters = filters
        self.depth = len(filters)

        assert len(isotropy) == self.depth
        if is_isotropic:
            isotropy = [True] * self.depth
        self.isotropy = isotropy

        self.shared_kwargs = {
            'pad_mode': pad_mode,
            'act_mode': act_mode,
            'norm_mode': norm_mode
        }

        backbone_kwargs = {
            'block_type': block_type,
            'in_channel': in_channel,
            'filters': filters,
            'isotropy': isotropy,
            'blocks': blocks,
            'deploy': deploy,
            'fmap_size': fmap_size,
            'ks': ks,
            'attention': attn,
        }
        backbone_kwargs.update(self.shared_kwargs)

        return_layers = {
            'layer0': feature_keys[0],
            'layer1': feature_keys[1],
            'layer2': feature_keys[2],
            'layer3': feature_keys[3],
            'layer4': feature_keys[4]
        }

        self.backbone = IntermediateLayerGetter(Effnet3D(**backbone_kwargs), return_layers)
        self.feature_keys = feature_keys

        self.middle_convs = nn.Sequential(
            conv3d_norm_act(self.filters[-1], self.filters[-1], **self.shared_kwargs),
            conv3d_norm_act(self.filters[-1], self.filters[-1], **self.shared_kwargs)
        )

        self.latplanes = filters[0]
        # self.latlayers = nn.ModuleList([
        #     conv3d_norm_act(x, self.latplanes, kernel_size=1, padding=0,
        #                     **self.shared_kwargs) for x in filters])

        self.smooth = nn.ModuleList()
        self.convs = nn.ModuleList()
        for i in range(self.depth-1):
            kernel_size, padding = self._get_kernel_size(isotropy[i])
            self.smooth.append(conv3d_norm_act(
                self.filters[i+1], self.filters[i], kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs))
            self.convs.append(conv3d_norm_act(
                self.filters[i]*2, self.filters[i], kernel_size=kernel_size,
                padding=padding, **self.shared_kwargs))

        self.post_up_conv = conv3d_norm_act(
                self.filters[0], self.filters[0], kernel_size=3,
                padding=1, **self.shared_kwargs)
        self.res_block = InvertedResidual(self.filters[0]+1, self.filters[0], 5, 1,
                                          isotropic=True, bias=True, **self.shared_kwargs)
        self.conv_out = self._get_io_conv(out_channel, isotropy[0])

        # initialization
        # model_init(self, init_mode)

    def forward(self, x):
        z = self.backbone(x)
        return self._forward_main(z, x)

    def _forward_main(self, z, x):
        # features = [self.latlayers[i](z[self.feature_keys[i]])
        #             for i in range(self.depth)]

        features = [z[self.feature_keys[i]] for i in range(self.depth)]
        out = self.middle_convs(features[self.depth-1])

        for i in range(self.depth-2, -1, -1):
            out = self._smooth_cat_conv(out, features[i], self.smooth[i], self.convs[i])
        
        # last cross
        cat = torch.cat(
            [x,
             self.post_up_conv(
                F.interpolate(out, size=x.shape[2:], mode='trilinear',
                          align_corners=True))],
            dim=1
        )
        out = self.res_block(cat) # + out
        out = self.conv_out(out)

        return out

    def _smooth_cat_conv(self, x, y, smooth, conv):
        """Upsample, smooth and add two feature maps.
        """
        x = F.interpolate(x, size=y.shape[2:], mode='trilinear',
                          align_corners=True)
        return conv(torch.cat([smooth(x), y], dim=1))

    def _get_kernel_size(self, is_isotropic, io_layer=False):
        if io_layer:  # kernel and padding size of I/O layers
            if is_isotropic:
                return (1, 1, 1), (0, 0, 0)
            return (1, 5, 5), (0, 2, 2)

        if is_isotropic:
            return (3, 3, 3), (1, 1, 1)
        return (1, 3, 3), (0, 1, 1)

    def _get_io_conv(self, out_channel, is_isotropic):
        kernel_size_io, padding_io = self._get_kernel_size(
            is_isotropic, io_layer=True)
        return conv3d_norm_act(
            self.filters[0], out_channel, kernel_size_io, padding=padding_io,
            pad_mode=self.shared_kwargs['pad_mode'], bias=True,
            act_mode='none', norm_mode='none')


class Effnet3D(nn.Module):
    expansion_factor = 1
    dilation_factors = [1, 2, 4, 8]
    num_stages = 5

    block_dict = {
        'inverted_res': InvertedResidual,
        'inverted_res_dilated': InvertedResidualDilated,
    }

    def __init__(self,
                 block_type: str = 'inverted_res',
                 in_channel: int = 1,
                 filters: List[int] = [32, 64, 96, 128, 160],
                 blocks: List[int] = [1, 2, 2, 2, 4],
                 ks: List[int] = [3, 3, 5, 3, 3],
                 isotropy: List[bool] = [False, False, False, True, True],
                 attention: str = 'squeeze_excitation',
                 bn_momentum: float = 0.01,
                 conv_type: str = 'standard',
                 pad_mode: str = 'replicate',
                 act_mode: str = 'elu',
                 norm_mode: str = 'bn',
                 **_):
        super().__init__()

        block = self.block_dict[block_type]
        self.inplanes = filters[0]

        self.conv1 = get_conv(conv_type)(
            in_channel,
            self.inplanes,
            kernel_size=(7,7,7),
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            padding_mode=pad_mode,
            bias=False)

        self.bn1 = get_norm_3d(norm_mode, self.inplanes, bn_momentum)
        self.relu = get_activation(act_mode)

        shared_kwargs = {
            'expansion_factor': self.expansion_factor,
            'bn_momentum': bn_momentum,
            'norm_mode': norm_mode,
            'attention': attention,
            'pad_mode': pad_mode,
            'act_mode': act_mode,
        }

        self.layer0 = dw_stack(block, filters[0], filters[0], kernel_size=ks[0], stride=(1, 2, 2),
                               repeats=blocks[0], isotropic=isotropy[0], shared=shared_kwargs)
        self.layer1 = dw_stack(block, filters[0], filters[1], kernel_size=ks[1], stride=(1, 2, 2),
                               repeats=blocks[1], isotropic=isotropy[1], shared=shared_kwargs)
        self.layer2 = dw_stack(block, filters[1], filters[2], kernel_size=ks[2], stride=1,
                               repeats=blocks[2], isotropic=isotropy[2], shared=shared_kwargs)
        self.layer3 = dw_stack(block, filters[2], filters[3], kernel_size=ks[3], stride=(2, 2, 2),
                               repeats=blocks[3], isotropic=isotropy[3], shared=shared_kwargs)
        self.layer4 = dw_stack(block, filters[3], filters[4], kernel_size=ks[4], stride=(2, 2, 2),
                               repeats=blocks[4], isotropic=isotropy[4], shared=shared_kwargs)

    def forward(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def dw_stack(block, in_ch, out_ch, kernel_size, stride,
             repeats, isotropic, shared):
    """ Creates a stack of inverted residual blocks. 
    """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = block(in_ch, out_ch, kernel_size, stride,
                  isotropic=isotropic, **shared)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            block(out_ch, out_ch, kernel_size, 1,
                  isotropic=isotropic, **shared))
    return nn.Sequential(first, *remaining)

