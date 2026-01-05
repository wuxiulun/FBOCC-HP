import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models import HEADS
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
from mmcv.runner import force_fp32, auto_fp16
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2

@HEADS.register_module()
class SimpleMLPHardnessNet(nn.Module):
    """
    最简单的MLP版本，每个体素独立处理
    """

    def __init__(self, input_dim=80, hidden_dims=[64, 32, 16]):
        super(SimpleMLPHardnessNet, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    @force_fp32()
    def forward(self, voxel_features):
        """
        前向传播 - 每个体素特征独立通过MLP
        输入: voxel_features [B, C, H, W, D]
        输出: hardness [B, H, W, D]
        """
        B, C, H, W, D = voxel_features.shape

        x = voxel_features.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(-1, C)

        hardness_flat = self.mlp(x)

        hardness = hardness_flat.view(B, H, W, D)

        return hardness