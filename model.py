# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class CA(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CA, self).__init__()
        self.channel = channel

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.channel, self.channel // ratio, True),
            nn.ReLU(),
            nn.Linear(self.channel // ratio, self.channel, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        inputs = x
        avg_ = self.avg_pool(x).view([b, c])
        max_ = self.max_pool(x).view([b, c])

        avg_fc_ = self.fc(avg_).view([b, c, 1, 1])
        max_fc_ = self.fc(max_).view([b, c, 1, 1])

        output = avg_fc_ + max_fc_

        return output * inputs


class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()
        assert kernel_size in (3, 7, 15, 27, 31)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        inputs = x
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.conv1(x)
        return inputs * self.sigmoid(x)


class SENET(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SENET, self).__init__()
        self.channel = channel

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 定义全局平均池化层，结果为[b, a, 1, 1] b表示batchsize, c表示通道数
        # 定义全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, True),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        inputs = x

        inpust_avg = self.avg_pool(x).view([b, c])

        inputs_fc = self.fc(inpust_avg).view([b, c, 1, 1])

        return inputs_fc * inputs


class ECANET(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECANET, self).__init__()
        self.channel = channel

        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = (kernel_size - 1) // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1d(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class CASA(nn.Module):
    def __init__(self, channel):
        super(CASA, self).__init__()
        self.CA = CA(channel)
        self.SA = SA(kernel_size=7)

    def forward(self, x):
        inputs = x
        ca_x = self.CA(x)
        output = ca_x * inputs
        sa_x = self.SA(output)
        output_ = sa_x * output

        return output_

class SACA(nn.Module):
    def __init__(self, channel):
        super(SACA, self).__init__()
        self.CA = CA(channel)
        self.SA = SA(kernel_size=7)

    def forward(self, x):
        inputs = x
        sa_x = self.SA(x)
        output = sa_x * inputs
        ca_x = self.CA(output)
        output_ = ca_x * output

        return output_



class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtAa(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1., aa="ca"
                 ):
        super().__init__()
        if aa == "ca":
            print("====> based on Channel Attention!")
            self.AA = CA(channel=dims[-1])
        elif aa == "sa":
            print("====> based on Spatial Attention!")
            self.AA = SA(kernel_size=3)
        elif aa == "senet":
            print("====> based on SENet!")
            self.AA = SENET(channel=dims[-1])
        elif aa == "ecanet":
            print("====> based on ECANet!")
            self.AA = ECANET(channel=dims[-1])
        elif aa == "casa":
            print("====> based on Channel Attention and Spatial Attention!")
            self.AA = CASA(channel=dims[-1])
        elif aa == "saca":
            print("====> based on Spatial Attention and Channel Attention!")
            self.AA = SACA(channel=dims[-1])
        elif aa == "fcanet":
            print("====> based on Frequency Attention and Channel Attention!")
            # self.AA = FcaLayer(channel=dims[-1],reduction=16,width=224,height=224)
            print(dims[-1])
        else:
            raise ValueError

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x + self.AA(x)
        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x

    def freeze_backbone(self):
        backbone = [self.downsample_layers, self.stages]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = False
            except:
                module.requires_grad = False

    def unfreeze_backbone(self):
        backbone = [self.downsample_layers, self.stages]
        for module in backbone:
            try:
                for param in module.parameters():
                    param.requires_grad = True
            except:
                module.requires_grad = True


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def convnextaa_base(num_classes: int, pretrained=True, aa="ca",path=None):
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth
    # https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth
    model = ConvNeXtAa(depths=[3, 3, 27, 3],
                       dims=[128, 256, 512, 1024],
                       num_classes=num_classes, aa=aa)
    if pretrained:
        # model.load_state_dict(torch.load("/data/CNX_V3/model_data/convnext_base_1k_224_ema.pth")["model"])
        # weights_dict = torch.load("/data/CNX_ImageNet100/model_data/convnext_base_1k_224_ema.pth")["model"]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         weights_dict = torch.load("../nets/cnxECANet-least_valloss.pth",map_location=device)
        weights_dict = torch.load(path,map_location=device)

#         删除有关分类类别的权重
#         for k in list(weights_dict.keys()):
#             if "head" in k:
#                 del weights_dict[k]
#             elif "norm.weight" == k:
#                 del weights_dict[k]
#             elif "norm.bias" == k:
#                 del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)
    return model
