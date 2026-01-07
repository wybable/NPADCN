"""
@File  : HsiConfermer.py
@Author: Able_WYB
@Date  : 2022/5/30 14:59
@Desc  : 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from activation import mish, gelu, gelu_new, swish
from functools import partial
from timm.models.layers import DropPath, trunc_normal_
from dcn.modules.deform_conv import DeformConv, DeformConv_d, DeformConvPack, DeformConvPack_d
import math
from torch.nn.functional import linear, normalize
from activation import mish


class MLPLayer(nn.Module):
    r"""
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
    """

    def __init__(self, in_features, out_features=None, hidden_features=None, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class simam_module_3d(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module_3d, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, C, H, W, B = x.shape

        n = H * W * B - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class simam_module_2d(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module_2d, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, C, H, W, B = x.shape

        n = H * W - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class simam_module_1d(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module_1d, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, C, H, W, B = x.shape

        n = B - 1

        x_minus_mu_square = (x - x.mean(dim=[4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[4], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)


class similarity_attention(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(similarity_attention, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "similarity_attention"

    def forward(self, x):
        b, C, H, W, B = x.shape
        assert W == H
        x = x.permute(0, 4, 2, 3, 1)
        center = H // 2
        cos_sim = F.cosine_similarity(x, x[:, :, center, center, :][:, :, None, None, :], dim=4)
        cos_sim = (cos_sim + 1) / 2
        out = cos_sim.unsqueeze(-1).expand(b, C, H, W, B)
        return out


class sim_similarity_3D(nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(sim_similarity_3D, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "similarity_attention"

    def forward(self, x):
        b, C, H, W, B = x.shape
        assert W == H
        center = H // 2
        cos_sim = F.cosine_similarity(x, x[:, :, center, center, :][:, :, None, None, :], dim=4)
        cos_sim = (cos_sim + 1) / 2
        cos_sim = cos_sim.unsqueeze(-1)
        n = B - 1
        x_minus_mu_square = (x - x.mean(dim=[4], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[4], keepdim=True) / n + self.e_lambda)) + 0.5
        attn_1d = self.activaton(y)
        attn_3d = attn_1d * cos_sim
        return attn_3d


class spa_defromconv_block(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch):
        super(spa_defromconv_block, self).__init__()
        # self.deform_conv1 = DeformConvPack_d(in_ch, mid_ch, kernel_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0],
        #                                      dimension='TH')
        self.deform_conv1 = nn.Conv3d(in_ch, mid_ch, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), )
        self.norm_1 = nn.Sequential(nn.BatchNorm3d(in_ch, eps=0.001, momentum=0.1, affine=True),
                                    mish())  # 动量默认值为0.1
        # self.deform_conv2 = DeformConvPack_d(mid_ch, out_ch, kernel_size=[3, 3, 1], stride=[1, 1, 1], padding=[1, 1, 0],
        #                                      dimension='TH')
        self.deform_conv2 = nn.Conv3d(mid_ch, out_ch, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0), )
        self.norm_2 = nn.Sequential(nn.BatchNorm3d(mid_ch, eps=0.001, momentum=0.1, affine=True),
                                    mish())  # 动量默认值为0.1
        if in_ch != out_ch:
            self.conv_one = nn.Conv3d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        _, c_x, _, _, _ = x.shape
        out = self.norm_1(x)
        out = self.deform_conv1(out)
        out = self.norm_2(out)
        out = self.deform_conv2(out)
        _, c_out, _, _, _ = out.shape
        if c_x != c_out:
            x = self.conv_one(x)
        return x + out


class spe_defromconv_block(nn.Module):
    def __init__(self, in_ch, out_ch, bands):
        super(spe_defromconv_block, self).__init__()
        self.norm_1 = nn.Sequential(nn.BatchNorm3d(in_ch, eps=0.001, momentum=0.1, affine=True),
                                    mish())  # 动量默认值为0.1
        # self.deform_conv1 = DeformConvPack_d(in_ch, out_ch, kernel_size=[1, 1, 3], stride=[1, 1, 1], padding=[0, 0, 1],
        #                                      dimension='W')
        self.conv1 = nn.Conv3d(in_ch, out_ch, (1, 1, 3), padding=(0, 0, 1))
        self.norm_2 = nn.LayerNorm(bands)
        self.act_2 = mish()
        self.mlp = MLPLayer(in_features=bands, out_features=bands, hidden_features=bands // 4)
        if in_ch != out_ch:
            self.conv_one_1 = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        out = self.norm_1(x)
        # out = self.deform_conv1(out)
        out = self.conv1(out)
        if x.shape[1] != out.shape[1]:
            x = self.conv_one_1(x)
        out = x + out
        x = out
        out = self.norm_2(out)
        out = self.act_2(out)
        out = self.mlp(out)
        return x + out


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Deform_net(nn.Module):
    def __init__(self, bands, num_classes=16, cnn_channel=16, ):
        super().__init__()
        self.name = 'Deform_net'
        self.bands = bands
        self.num_classes = num_classes
        self.cnn_channel = cnn_channel

        #  降维操作
        self.conv_stem_1 = nn.Conv3d(1, self.cnn_channel, kernel_size=(1, 1, 7), stride=(1, 1, 3), padding=(0, 0, 3), )
        self.norm_stem_1 = nn.Sequential(nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
                                         mish())
        self.conv_stem_2 = nn.Conv3d(1, self.cnn_channel, kernel_size=(1, 1, bands), stride=1, )
        self.norm_stem_2 = nn.Sequential(nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
                                         mish())

        #  无参数注意力机制定义
        # self.attn_3d = simam_module_3d()
        self.attn_1d = simam_module_1d()
        self.attn_similarity = similarity_attention()
        self.attn_3d_sim = sim_similarity_3D()

        #  空间特征提取模块
        self.spa_layer_1_attenconv = nn.Conv3d(self.cnn_channel, 2 * 3 * 3, kernel_size=(3, 3, 1),
                                               stride=(1, 1, 1), padding=(1, 1, 0), )
        self.spa_layer_1_deconv = DeformConv_d(self.cnn_channel, self.cnn_channel, kernel_size=[3, 3, 1],
                                               stride=[1, 1, 1], padding=[1, 1, 0], dimension='TH')
        self.spa_layer_1_norm = nn.Sequential(
            nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.spa_layer_2_attenconv = nn.Conv3d(2 * self.cnn_channel, 2 * 3 * 3, kernel_size=(3, 3, 1),
                                               stride=(1, 1, 1), padding=(1, 1, 0))
        self.spa_layer_2_deconv = DeformConv_d(2 * self.cnn_channel, self.cnn_channel, kernel_size=[3, 3, 1],
                                               stride=[1, 1, 1], padding=[1, 1, 0], dimension='TH')
        self.spa_layer_2_norm = nn.Sequential(
            nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        # self.spa_layer = nn.Sequential(
        #     nn.Conv3d(3 * self.cnn_channel, 3 * self.cnn_channel, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
        #     nn.BatchNorm3d(3 * self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
        #     mish()
        # )

        #  光谱特征提取模块
        self.spe_layer_1_attenconv = nn.Conv3d(self.cnn_channel, 1 * 3, kernel_size=(1, 1, 3),
                                               stride=(1, 1, 1), padding=(0, 0, 1))
        self.spe_layer_1_deconv = DeformConv_d(self.cnn_channel, self.cnn_channel, kernel_size=[1, 1, 3],
                                               stride=[1, 1, 1], padding=[0, 0, 1], dimension='W')
        self.spe_layer_1_norm = nn.Sequential(
            nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        self.spe_layer_2_attenconv = nn.Conv3d(2 * self.cnn_channel, 1 * 3, kernel_size=(1, 1, 3),
                                               stride=(1, 1, 1), padding=(0, 0, 1))
        self.spe_layer_2_deconv = DeformConv_d(2 * self.cnn_channel, self.cnn_channel, kernel_size=[1, 1, 3],
                                               stride=[1, 1, 1], padding=[0, 0, 1], dimension='W')
        self.spe_layer_2_norm = nn.Sequential(
            nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.spe_bands = bands // 3
        if bands % 3 != 0:
            self.spe_bands = self.spe_bands + 1
        self.spe_layer = nn.Sequential(
            nn.Conv3d(3 * self.cnn_channel, 3 * self.cnn_channel, kernel_size=(1, 1, self.spe_bands)),
            nn.BatchNorm3d(3 * self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )

        # 空谱特征联合提取
        # self.spa_spe_layer_1 = DeformConvPack(1, self.cnn_channel, kernel_size=[3, 3, 3],
        #                                       stride=[1, 1, 1], padding=[1, 1, 1])
        # self.spa_spe_layer_2 = nn.Sequential(
        #     nn.BatchNorm3d(self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
        #     mish()
        # )

        self.spa_spe_layer = nn.Sequential(

            zero_module(nn.Conv3d(1, 6 * self.cnn_channel, kernel_size=(3, 3, 6 * self.cnn_channel), padding=(1, 1, 0))),
            # zero_module(nn.Conv3d(self.cnn_channel, 6 * self.cnn_channel, kernel_size=(1, 1, 6 * self.cnn_channel), padding=(0, 0, 0))),
            nn.BatchNorm3d(6 * self.cnn_channel, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.out_pool = nn.AdaptiveAvgPool3d(1)
        self.out_classes = nn.Linear(6 * self.cnn_channel, self.num_classes)
        self._init_weights(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x = x.permute(0, 1, 3, 4, 2)  # 如果x.shape = b, C, B, H, W
        b, C, H, W, B = x.shape  # [B, 1, 9, 9, 200,]

        x11 = self.conv_stem_1(x)  # [B, 1, 9, 9, 200,] -> [B, 16, 9, 9, 100]
        x11 = self.norm_stem_1(x11)

        x11_atten = self.attn_1d(x11)
        x11_offset = self.spe_layer_1_attenconv(x11_atten)
        x11 = x11_atten * x11
        x12 = self.spe_layer_1_deconv(x11, x11_offset)
        x12 = self.spe_layer_1_norm(x12)
        x12_atten = self.attn_1d(x12)
        x12_offset = self.spe_layer_2_attenconv(torch.concat([x11_atten, x12_atten], dim=1))
        x12 = x12_atten * x12
        x13 = self.spe_layer_2_deconv(torch.concat([x11, x12], dim=1), x12_offset)
        x13 = self.spe_layer_2_norm(x13)
        spe_features = torch.concat([x11, x12, x13], dim=1)
        spe_features = self.spe_layer(spe_features)

        x21 = self.conv_stem_2(x)  # [B, 1, 9, 9, 200,] -> [B, 64, 9, 9, 1]
        x21 = self.norm_stem_2(x21)

        x21_atten = self.attn_similarity(x21)
        x21_offset = self.spa_layer_1_attenconv(x21_atten)
        x21 = x21_atten * x21
        x22 = self.spa_layer_1_deconv(x21, x21_offset)
        x22 = self.spa_layer_1_norm(x22)
        x22_atten = self.attn_similarity(x22)
        x22_offset = self.spa_layer_2_attenconv(torch.concat([x21_atten, x22_atten], dim=1))
        x22 = x22_atten * x22
        x23 = self.spa_layer_2_deconv(torch.concat([x21, x22], dim=1), x22_offset)
        x23 = self.spa_layer_2_norm(x23)
        spa_features = torch.concat([x21, x22, x23], dim=1)
        # spa_features = self.spa_layer(spa_features)

        features = torch.concat([spa_features, spe_features], dim=1).permute(0, 4, 2, 3, 1)
        atten = self.attn_3d_sim(features)
        features = atten * features
        # features = self.spa_spe_layer_1(features, atten)
        # features = self.spa_spe_layer_2(features)
        features = self.spa_spe_layer(features)
        out = self.out_pool(features).squeeze()
        out = self.out_classes(out)
        return out
