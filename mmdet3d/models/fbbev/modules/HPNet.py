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


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    @force_fp32()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    @force_fp32()
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    @force_fp32()
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    @force_fp32()
    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


@HEADS.register_module()
class DepthAwareHardnessNet(BaseModule):
    """
    深度感知的困难区域预测网络
    输入: depth (1,6,80,16,44) 和 context (1,6,80,16,44)
    输出: hardness (1,6,80,16,44)
    """

    def __init__(self,
                 depth_channels=80,  # 深度通道数
                 context_channels=80,  # 上下文特征通道数
                 mid_channels=512,  # 中间通道数
                 use_aspp=True,  # 使用多尺度感受野
                 use_3d_conv=False,  # 是否使用3D卷积处理深度维度
                 use_geometry_aware=True,  # 使用几何感知
                 with_cp=False,  # 梯度检查点
                 temperature=0.1):  # 分布温度参数
        super(DepthAwareHardnessNet, self).__init__()

        self.depth_channels = depth_channels
        self.context_channels = context_channels
        self.mid_channels = mid_channels
        self.temperature = temperature
        # self.with_cp = False
        self.with_cp = with_cp
        self.use_geometry_aware = use_geometry_aware
        self.use_3d_conv = use_3d_conv

        # 1. 特征融合模块 - 将depth和context信息融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(depth_channels + context_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 深度维度处理模块
        if use_3d_conv:
            # 使用3D卷积处理深度维度
            self.depth_processor = nn.Sequential(
                nn.Conv3d(mid_channels, mid_channels, (3, 3, 3), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_channels, mid_channels, (3, 3, 3), padding=(1, 1, 1), bias=False),
                nn.BatchNorm3d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            # 使用2D卷积，但在通道维度包含深度信息
            self.depth_processor = nn.Sequential(
                BasicBlock(mid_channels, mid_channels),
                BasicBlock(mid_channels, mid_channels),
            )

        # 3. 几何感知模块
        if use_geometry_aware:
            self.bn = nn.BatchNorm1d(27)
            self.hardness_mlp = Mlp(27, mid_channels, mid_channels)
            self.hardness_se = SELayer(mid_channels)

        # 4. 多尺度上下文提取
        conv_list = []
        if use_aspp:
            conv_list.append(ASPP(mid_channels, mid_channels))

        conv_list.extend([
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        ])

        self.context_extractor = nn.Sequential(*conv_list)

        # 5. 输出模块 - 生成每个深度位置的困难度
        self.output_conv = nn.Sequential(
            nn.Conv2d(mid_channels // 2, depth_channels, 1),  # 输出深度通道数的困难度
            nn.Sigmoid()  # 输出0-1范围的困难度
        )

        # 6. 初始化权重
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @force_fp32()
    def forward(self, depth, context, mlp_input=None):
        """
        前向传播
        输入:
            depth: 深度预测置信度 [B, N, D, H, W] = [1, 6, 80, 16, 44]
            context: 图像特征 [B, N, C, H, W] = [1, 6, 80, 16, 44]
            mlp_input: 相机参数 [B, N, 27] (可选)
        输出:
            hardness: 每个深度位置的困难度 [B, N, D, H, W] = [1, 6, 80, 16, 44]
            hardness_prob: 用于监督的概率分布 [B, N, D, H, W]
        """
        B, N, D, H, W = depth.shape
        C = context.shape[2]  # 应该是80

        # 重塑为2D特征图格式
        depth_2d = depth.view(B * N, D, H, W)  # [6, 80, 16, 44]
        context_2d = context.view(B * N, C, H, W)  # [6, 80, 16, 44]

        # # 1. 特征融合 - 添加梯度检查点
        # if self.with_cp and torch.is_grad_enabled():
        #     fused_feat = cp.checkpoint(self.feature_fusion, torch.cat([depth_2d, context_2d], dim=1))
        # else:
        fused_feat = self.feature_fusion(torch.cat([depth_2d, context_2d], dim=1))  # [6, 256, 16, 44]

        # 2. 深度维度处理 - 添加梯度检查点
        if self.use_3d_conv:
            # 将特征重塑为3D格式 [B*N, C, D, H, W]
            fused_3d = fused_feat.unsqueeze(2).repeat(1, 1, D, 1, 1)  # [6, 256, 80, 16, 44]
            if self.with_cp and torch.is_grad_enabled():
                processed_feat = cp.checkpoint(self.depth_processor, fused_3d)
            else:
                processed_feat = self.depth_processor(fused_3d)  # [6, 256, 80, 16, 44]
            # 压缩通道维度，准备后续处理
            processed_feat = processed_feat.mean(dim=2)  # [6, 256, 16, 44]
        else:
            if self.with_cp and torch.is_grad_enabled():
                processed_feat = cp.checkpoint(self.depth_processor, fused_feat)
            else:
                processed_feat = self.depth_processor(fused_feat)  # [6, 256, 16, 44]

        # 3. 几何感知调制
        if self.use_geometry_aware and mlp_input is not None:
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            hardness_se = self.hardness_mlp(mlp_input)[..., None, None]
            if self.with_cp and torch.is_grad_enabled():
                processed_feat = cp.checkpoint(self.hardness_se, processed_feat, hardness_se)
            else:
                processed_feat = self.hardness_se(processed_feat, hardness_se)

        # 4. 多尺度上下文提取 - 添加梯度检查点
        if self.with_cp and torch.is_grad_enabled():
            context_feat = cp.checkpoint(self.context_extractor, processed_feat)
        else:
            context_feat = self.context_extractor(processed_feat)  # [6, 128, 16, 44]

        # 5. 生成每个深度位置的困难度
        hardness_2d = self.output_conv(context_feat)  # [6, 80, 16, 44]

        # 6. 调整输出形状
        hardness = hardness_2d.view(B, N, D, H, W)  # [1, 6, 80, 16, 44]

        return hardness
