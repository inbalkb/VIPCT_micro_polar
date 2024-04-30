# This file contains the code of the adapted Resnet50-FPN image feature extractor in VIP-CT.
# You are very welcome to use this code. For this, clearly acknowledge
# the source of this code, and cite the paper described in the readme file:
# Roi Ronen, Vadim Holodovsky and Yoav. Y. Schechner, "Variable Imaging Projection Cloud Scattering Tomography",
# Proc. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
#
# Copyright (c) Roi Ronen. The python code is available for
# non-commercial use and exploration.  For commercial use contact the
# author. The author is not liable for any damages or loss that might be
# caused by use or connection to this code.
# All rights reserved.
#
# This source code is licensed under the Apache License 2.0 found in the
# LICENSE file in the root directory of this source tree.

from torchvision.models import resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool
from torch import nn
from torchvision.models.detection.backbone_utils import BackboneWithFPN
import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional

def resnet34_with_env_backbone(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> resnet.ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', resnet.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def _resnet(
    arch: str,
    block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any):
    model = resnet_with_env_backbone(block, layers, **kwargs)
    return model

class resnet_with_env_backbone(resnet.ResNet):
    def __init__(
        self,
        block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        in_channels: int = 3,
        in_env_channels: int = 3,
        num_layers: int = 4,
        use_first_pool: bool = False
    ) -> None:
        super(resnet.ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_layers = num_layers
        self.use_first_pool = use_first_pool
        levels_out_channels = [64, 64, 128, 256, 512]
        self.inplanes = levels_out_channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, levels_out_channels[1], layers[0])
        self.layer2 = self._make_layer(block, levels_out_channels[2], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, levels_out_channels[3], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, levels_out_channels[4], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        insert_env_layers = [nn.Conv2d(levels_out_channels[self.num_layers-2]+in_env_channels, levels_out_channels[self.num_layers-2],
                                       kernel_size=1, stride=1, padding=0,bias=False),
                             norm_layer(levels_out_channels[self.num_layers-2]), nn.ReLU(inplace=True)]
        self.insert_env = nn.Sequential(*insert_env_layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[resnet.BasicBlock, resnet.Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, env_params: Tensor) -> Tensor:
        input_size = torch.tensor(x.shape[-2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        latents = [x]
        if self.num_layers == 2:
            env_in = env_params[:, :, None, None].tile(x.shape[0], 1, x.shape[2], x.shape[3])
            x = torch.cat([x, env_in], dim=1)
            x = self.insert_env(x)
        if self.num_layers > 1:
            if self.use_first_pool:
                x = self.maxpool(x)
            x = self.layer1(x)
            latents.append(x)
        if self.num_layers == 3:
            env_in = env_params[:, :, None, None].tile(x.shape[0], 1, x.shape[2], x.shape[3])
            x = torch.cat([x, env_in], dim=1)
            x = self.insert_env(x)
        if self.num_layers > 2:
            x = self.layer2(x)
            latents.append(x)
        if self.num_layers == 4:
            env_in = env_params[:, :, None, None].tile(1, 1, x.shape[2], x.shape[3])
            x = torch.cat([x, env_in], dim=1)
            x = self.insert_env(x)
        if self.num_layers > 3:
            x = self.layer3(x)
            latents.append(x)
        if self.num_layers == 5:
            env_in = env_params[:, :, None, None].tile(x.shape[0], 1, x.shape[2], x.shape[3])
            x = torch.cat([x, env_in], dim=1)
            x = self.insert_env(x)
        if self.num_layers > 4:
            x = self.layer4(x)
            latents.append(x)

        self.latent_scaling = [(torch.tensor(latent.shape[-2:]) / input_size).to(device=x.device) for latent in latents]
        # return latent
        return latents


    def forward(self, x: Tensor, env_params: Tensor) -> Tensor:
        return self._forward_impl(x, env_params)


