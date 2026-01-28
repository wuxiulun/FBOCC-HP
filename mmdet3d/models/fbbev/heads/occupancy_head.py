# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet3d.models.fbbev.modules.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmdet3d.models.fbbev.modules.occ_loss_utils import nusc_class_frequencies, nusc_class_names
from mmdet3d.models.fbbev.modules.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from torch.utils.checkpoint import checkpoint as cp
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp import autocast
from mmdet3d.models import builder
from mmdet3d.models.fbbev.modules.occ_loss_utils import LossDistributionCalculator, HPNSupervisionLoss

@HEADS.register_module()
class OccHead(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channel,
        num_level=1,
        soft_weights=False,
        loss_weight_cfg=None,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        final_occ_size=[256, 256, 20],
        empty_idx=0,
        balance_cls_weight=True,
        train_cfg=None,
        test_cfg=None,
        with_cp=False,
        use_focal_loss=False,
        use_dice_loss= False,
        use_deblock=True,
    ):
        super(OccHead, self).__init__()

        self.fp16_enabled=False
      
        if type(in_channels) is not list:
            in_channels = [in_channels]
        self.with_cp = with_cp
        self.use_deblock = use_deblock
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        
        self.point_cloud_range = torch.tensor(np.array(point_cloud_range)).float()

        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        


        # voxel-level prediction
        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i], 
                        out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True))
            self.occ_convs.append(occ_conv)


        self.occ_pred_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                        out_channels=mid_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, mid_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel//2, 
                        out_channels=out_channel, kernel_size=1, stride=1, padding=0))

        self.soft_weights = soft_weights
        self.num_point_sampling_feat = self.num_level + 1 * self.use_deblock
        if self.soft_weights:
            soft_in_channel = mid_channel
            self.voxel_soft_weights = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=soft_in_channel, 
                        out_channels=soft_in_channel//2, kernel_size=1, stride=1, padding=0),
                build_norm_layer(norm_cfg, soft_in_channel//2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=soft_in_channel//2, 
                        out_channels=self.num_point_sampling_feat, kernel_size=1, stride=1, padding=0))
            
        # loss functions
        self.use_dice_loss = use_dice_loss
        if self.use_dice_loss:
            self.dice_loss = builder.build_loss(dict(type='DiceLoss', loss_weight=2))

        if balance_cls_weight:
            if out_channel == 19:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
            else:
                if out_channel == 17: nusc_class_frequencies[0] += nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:out_channel] + 0.001))
        else:
            self.class_weights = torch.ones(out_channel)/out_channel  # FIXME hardcode 17

        if self.use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=self.in_channels[0],
                    out_channels=self.in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, self.in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))


        self.class_names = nusc_class_names    
        self.empty_idx = empty_idx
    
    @force_fp32(apply_to=('voxel_feats')) 
    def forward_coarse_voxel(self, voxel_feats):
        output_occs = []
        output = {}

        if self.use_deblock:
            if self.with_cp and voxel_feats[0].requires_grad:
                x0 = cp(self.deblock, voxel_feats[0])
            else:
                x0 = self.deblock(voxel_feats[0])
            output_occs.append(x0)
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp  and feats.requires_grad:
                x = cp(occ_conv, feats)
            else:
                x = occ_conv(feats)
            output_occs.append(x)

        if self.soft_weights:
            voxel_soft_weights = self.voxel_soft_weights(output_occs[0])
            voxel_soft_weights = torch.softmax(voxel_soft_weights, dim=1)
        else:
            voxel_soft_weights = torch.ones([output_occs[0].shape[0], self.num_point_sampling_feat, 1, 1, 1],).to(output_occs[0].device) / self.num_point_sampling_feat

        out_voxel_feats = 0
        _, _, H, W, D= output_occs[0].shape
        for feats, weights in zip(output_occs, torch.unbind(voxel_soft_weights, dim=1)):
            feats = F.interpolate(feats, size=[H, W, D], mode='trilinear', align_corners=False).contiguous()
            out_voxel_feats += feats * weights.unsqueeze(1)
        output['out_voxel_feats'] = [out_voxel_feats]
        if self.with_cp and  out_voxel_feats.requires_grad:
            out_voxel = cp(self.occ_pred_conv, out_voxel_feats)
        else:
            out_voxel = self.occ_pred_conv(out_voxel_feats)

        output['occ'] = [out_voxel]

        return output
     
    @force_fp32()
    def forward(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, **kwargs):
        
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level
        
        output = self.forward_coarse_voxel(voxel_feats)
        out_voxel_feats = output['out_voxel_feats'][0]
        coarse_occ = output['occ'][0]

        res = {
            'output_voxels': output['occ'],
            'output_voxels_fine': output.get('fine_output', None),
            'output_coords_fine': output.get('fine_coord', None),
        }


        return res
    
    # @force_fp32()
    # def forward(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, **kwargs):

    #     assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

    #     output = self.forward_coarse_voxel(voxel_feats)
    #     out_voxel_feats = output['out_voxel_feats'][0]
    #     coarse_occ = output['occ'][0]

    #     # ==================== 新增：在测试阶段也计算统计和损失分布 ====================
    #     hardness_pred = kwargs.get('hardness_pred', None)
    #     scene_hardness_pred = kwargs.get('scene_hardness_pred', None)
    #     target_voxels = kwargs.get('gt_occupancy', None)
    #     visible_mask = kwargs.get('visible_mask', None)  # 可见性掩码

    #     # 如果target_voxels是list，取第一个
    #     if target_voxels is not None and isinstance(target_voxels, list):
    #         target_voxels = target_voxels[0]
        
    #     # 获取预测的空间维度
    #     B, C, H, W, D = coarse_occ.shape

    #     # 调整hardness_pred和scene_hardness_pred的形状以匹配occupancy
    #     if hardness_pred is not None:
    #         if hardness_pred.dim() == 3:  # [H, W] 或 [H, W, D]
    #             if hardness_pred.shape[-1] != D:  # 如果是2D的BEV特征
    #                 # 扩展到3D
    #                 hardness_pred = hardness_pred.unsqueeze(-1).expand(-1, -1, D)
    #             # 确保有batch维度
    #             if hardness_pred.dim() == 3:
    #                 hardness_pred = hardness_pred.unsqueeze(0)
    #         elif hardness_pred.dim() == 4:  # [B, H, W, D]
    #             pass  # 已经是正确形状
    #         else:
    #             # 调整到正确形状
    #             hardness_pred = F.interpolate(
    #                 hardness_pred.unsqueeze(1),
    #                 size=(H, W, D),
    #                 mode='trilinear',
    #                 align_corners=False
    #             ).squeeze(1)

    #     if scene_hardness_pred is not None:
    #         if scene_hardness_pred.dim() == 3:  # [H, W] 或 [H, W, D]
    #             if scene_hardness_pred.shape[-1] != D:
    #                 scene_hardness_pred = scene_hardness_pred.unsqueeze(-1).expand(-1, -1, D)
    #             if scene_hardness_pred.dim() == 3:
    #                 scene_hardness_pred = scene_hardness_pred.unsqueeze(0)
    #         elif scene_hardness_pred.dim() == 4:
    #             pass
    #         else:
    #             scene_hardness_pred = F.interpolate(
    #                 scene_hardness_pred.unsqueeze(1),
    #                 size=(H, W, D),
    #                 mode='trilinear',
    #                 align_corners=False
    #             ).squeeze(1)

    #     # 计算全局困难度（基于置信度）
    #     global_hardness = None
    #     if coarse_occ is not None:
    #         if coarse_occ.shape[1] == self.out_channel:
    #             logits_excluding_ignore = coarse_occ[:, 1:, :, :, :]
    #         else:
    #             logits_excluding_ignore = coarse_occ

    #         probs = F.softmax(logits_excluding_ignore, dim=1)
    #         top2_probs, _ = torch.topk(probs, k=2, dim=1)
    #         prob_diff = top2_probs[:, 0, ...] - top2_probs[:, 1, ...]
    #         epsilon = 1e-3
    #         prob_diff = torch.clamp(prob_diff, min=epsilon, max=1.0 - epsilon)
    #         global_hardness_raw = 1.0 / prob_diff
    #         global_hardness = torch.log(global_hardness_raw)  # [B, H, W, D]

    #     # 计算损失分布（如果有gt）
    #     loss_distribution = None
    #     monitor_stats = {}

    #     # ==================== 新增：计算visible_mask相关指标 ====================
    #     visible_monitor_stats = {}

    #     if target_voxels is not None:
    #         # 使用LossDistributionCalculator计算真实损失分布
    #         loss_calculator = LossDistributionCalculator(
    #             use_focal_loss=self.use_focal_loss,
    #             class_weights=self.class_weights,
    #             empty_idx=self.empty_idx
    #         )

    #         with torch.no_grad():
    #             loss_distribution, valid_mask = loss_calculator(coarse_occ.detach(), target_voxels)

    #         B_loss, H_true, W_true, D_true = loss_distribution.shape

    #         # 上采样hardness预测到损失分布的分辨率
    #         hardness_pred_upsampled = F.interpolate(
    #             hardness_pred.unsqueeze(1),
    #             size=(H_true, W_true, D_true),
    #             mode='trilinear',
    #             align_corners=False
    #         ).squeeze(1)

    #         scene_hardness_pred_upsampled = F.interpolate(
    #             scene_hardness_pred.unsqueeze(1),
    #             size=(H_true, W_true, D_true),
    #             mode='trilinear',
    #             align_corners=False
    #         ).squeeze(1)

    #         # ==================== 新增：处理visible_mask ====================
    #         if visible_mask is not None:
    #             # 调整visible_mask到正确的形状和分辨率
    #             if visible_mask.dim() == 3:  # [H, W, D] 或 [H, W]
    #                 if visible_mask.shape[-1] != D_true:  # 如果是2D
    #                     # 扩展到3D
    #                     visible_mask = visible_mask.unsqueeze(-1).expand(-1, -1, D_true)
    #                 # 确保有batch维度
    #                 if visible_mask.dim() == 3:
    #                     visible_mask = visible_mask.unsqueeze(0)
    #             elif visible_mask.dim() == 4:  # [B, H, W, D]
    #                 # 调整到正确的分辨率
    #                 visible_mask = F.interpolate(
    #                     visible_mask.float().unsqueeze(1),
    #                     size=(H_true, W_true, D_true),
    #                     mode='nearest'
    #                 ).squeeze(1).bool()
                
    #             # 计算visible_valid_mask：visible_mask和valid_mask的交集
    #             visible_valid_mask = visible_mask & valid_mask
                
    #             # 计算visible区域的基本统计
    #             visible_monitor_stats['visible_voxel_count'] = visible_valid_mask.sum().item()
    #             visible_monitor_stats['visible_ratio'] = visible_valid_mask.sum().item() / (valid_mask.sum().item() + 1e-8)
                
    #             # 只有在visible区域有有效体素时才计算visible指标
    #             if visible_valid_mask.sum() > 0:
    #                 # 获取visible区域的困难度和损失
    #                 visible_hardness = hardness_pred_upsampled[visible_valid_mask].flatten()
    #                 visible_scene_hardness = scene_hardness_pred_upsampled[visible_valid_mask].flatten()
    #                 visible_loss = loss_distribution[visible_valid_mask].flatten()
                    
    #                 # 计算visible区域的全局困难度
    #                 if global_hardness is not None:
    #                     global_hardness_upsampled = F.interpolate(
    #                         global_hardness.unsqueeze(1),
    #                         size=(H_true, W_true, D_true),
    #                         mode='trilinear',
    #                         align_corners=False
    #                     ).squeeze(1)
    #                     visible_global_hardness = global_hardness_upsampled[visible_valid_mask].flatten()
                    
    #                 # 计算visible区域的统计指标
    #                 n_visible = len(visible_loss)
                    
    #                 if n_visible > 0:
    #                     visible_monitor_stats['visible_loss_mean'] = visible_loss.mean().item()
    #                     visible_monitor_stats['visible_hardness_mean'] = visible_hardness.mean().item()
    #                     visible_monitor_stats['visible_scene_hardness_mean'] = visible_scene_hardness.mean().item()
                        
    #                     # ==================== 新增：计算visible区域的top1指标 ====================
    #                     # 前1%困难度对应的Loss均值对比（visible区域）
    #                     if n_visible >= 100:
    #                         k_1percent = max(1, n_visible // 100)
                            
    #                         # 获取前1%的索引
    #                         _, top1_hardness_idx = torch.topk(visible_hardness, k_1percent)
    #                         _, top1_scene_idx = torch.topk(visible_scene_hardness, k_1percent)
                            
    #                         # 计算对应的loss均值
    #                         top1_hardness_loss_mean = visible_loss[top1_hardness_idx].mean().item()
    #                         top1_scene_loss_mean = visible_loss[top1_scene_idx].mean().item()
    #                         overall_visible_loss_mean = visible_loss.mean().item()
                            
    #                         # 计算相对提升
    #                         visible_hardness_relative_gain = (top1_hardness_loss_mean - overall_visible_loss_mean) / overall_visible_loss_mean * 100
    #                         visible_scene_relative_gain = (top1_scene_loss_mean - overall_visible_loss_mean) / overall_visible_loss_mean * 100
                            
    #                         visible_monitor_stats['visible_top1_hardness_loss_mean'] = top1_hardness_loss_mean
    #                         visible_monitor_stats['visible_top1_scene_loss_mean'] = top1_scene_loss_mean
    #                         visible_monitor_stats['visible_hardness_relative_gain'] = visible_hardness_relative_gain
    #                         visible_monitor_stats['visible_scene_relative_gain'] = visible_scene_relative_gain
                            
    #                         # ==================== 新增：计算visible区域的top1 precision ====================
    #                         # 获取真实loss前1%的索引（在visible区域内）
    #                         _, top1_loss_idx = torch.topk(visible_loss, k_1percent)
                            
    #                         # 转换为集合
    #                         top1_hardness_set = set(top1_hardness_idx.cpu().numpy())
    #                         top1_scene_set = set(top1_scene_idx.cpu().numpy())
    #                         top1_loss_set = set(top1_loss_idx.cpu().numpy())
                            
    #                         # 计算交集
    #                         inter_hardness = len(top1_hardness_set & top1_loss_set)
    #                         inter_scene = len(top1_scene_set & top1_loss_set)
                            
    #                         # 计算精确率（precision）
    #                         visible_hardness_precision = inter_hardness / k_1percent
    #                         visible_scene_precision = inter_scene / k_1percent
                            
    #                         # 计算IoU
    #                         union_hardness = len(top1_hardness_set | top1_loss_set)
    #                         union_scene = len(top1_scene_set | top1_loss_set)
                            
    #                         visible_hardness_iou = inter_hardness / union_hardness if union_hardness > 0 else 0
    #                         visible_scene_iou = inter_scene / union_scene if union_scene > 0 else 0
                            
    #                         # 保存top1 precision指标
    #                         visible_monitor_stats['visible_hardness_precision'] = visible_hardness_precision
    #                         visible_monitor_stats['visible_scene_precision'] = visible_scene_precision
    #                         visible_monitor_stats['visible_hardness_iou'] = visible_hardness_iou
    #                         visible_monitor_stats['visible_scene_iou'] = visible_scene_iou
                            
    #                         if global_hardness is not None:
    #                             _, top1_global_idx = torch.topk(visible_global_hardness, k_1percent)
    #                             top1_global_loss_mean = visible_loss[top1_global_idx].mean().item()
    #                             visible_global_relative_gain = (top1_global_loss_mean - overall_visible_loss_mean) / overall_visible_loss_mean * 100
    #                             visible_monitor_stats['visible_top1_global_loss_mean'] = top1_global_loss_mean
    #                             visible_monitor_stats['visible_global_relative_gain'] = visible_global_relative_gain
                                
    #                             # ==================== 新增：计算visible区域的全局困难度top1 precision ====================
    #                             top1_global_set = set(top1_global_idx.cpu().numpy())
    #                             inter_global = len(top1_global_set & top1_loss_set)
    #                             visible_global_precision = inter_global / k_1percent
    #                             union_global = len(top1_global_set | top1_loss_set)
    #                             visible_global_iou = inter_global / union_global if union_global > 0 else 0
                                
    #                             visible_monitor_stats['visible_global_precision'] = visible_global_precision
    #                             visible_monitor_stats['visible_global_iou'] = visible_global_iou
                        
    #                     # 相关性计算（visible区域）
    #                     if len(visible_hardness) >= 2:
    #                         # 皮尔逊相关性
    #                         visible_hardness_corr_matrix = torch.corrcoef(torch.stack([visible_hardness, visible_loss]))
    #                         visible_hardness_corr = visible_hardness_corr_matrix[0, 1].item()
    #                         visible_monitor_stats['visible_hardness_pearson_corr'] = visible_hardness_corr
                            
    #                         visible_scene_corr_matrix = torch.corrcoef(torch.stack([visible_scene_hardness, visible_loss]))
    #                         visible_scene_corr = visible_scene_corr_matrix[0, 1].item()
    #                         visible_monitor_stats['visible_scene_pearson_corr'] = visible_scene_corr
                            
    #                         if global_hardness is not None:
    #                             visible_global_corr_matrix = torch.corrcoef(torch.stack([visible_global_hardness, visible_loss]))
    #                             visible_global_corr = visible_global_corr_matrix[0, 1].item()
    #                             visible_monitor_stats['visible_global_pearson_corr'] = visible_global_corr
                            
    #                         # ==================== 新增：计算visible区域的斯皮尔曼相关性 ====================
    #                         def spearman_correlation_efficient_direct(x, y):
    #                             n = x.shape[0]
    #                             if n < 2:
    #                                 return 0.0
    #                             x_ranks = torch.argsort(torch.argsort(x))
    #                             y_ranks = torch.argsort(torch.argsort(y))
    #                             corr_matrix = torch.corrcoef(torch.stack([x_ranks.float(), y_ranks.float()]))
    #                             return corr_matrix[0, 1].item()
                            
    #                         visible_monitor_stats['visible_hardness_spearman_corr'] = spearman_correlation_efficient_direct(visible_hardness, visible_loss)
    #                         visible_monitor_stats['visible_scene_spearman_corr'] = spearman_correlation_efficient_direct(visible_scene_hardness, visible_loss)
                            
    #                         if global_hardness is not None:
    #                             visible_monitor_stats['visible_global_spearman_corr'] = spearman_correlation_efficient_direct(visible_global_hardness, visible_loss)
            
    #         # 获取所有有效体素的困难度和损失（原有的逻辑）
    #         valid_hardness = hardness_pred_upsampled[valid_mask].flatten()
    #         valid_scene_hardness = scene_hardness_pred_upsampled[valid_mask].flatten()
    #         valid_loss = loss_distribution[valid_mask].flatten()

    #         if global_hardness is not None:
    #             global_hardness_upsampled = F.interpolate(
    #                 global_hardness.unsqueeze(1),
    #                 size=(H_true, W_true, D_true),
    #                 mode='trilinear',
    #                 align_corners=False
    #             ).squeeze(1)
    #             valid_global_hardness = global_hardness_upsampled[valid_mask].flatten()

    #         # 计算统计指标（原有的逻辑）
    #         n_total = len(valid_loss)

    #         if n_total > 0:
    #             monitor_stats['valid_voxel_count'] = n_total
    #             monitor_stats['overall_loss_mean'] = valid_loss.mean().item()
    #             monitor_stats['loss_range'] = [valid_loss.min().item(), valid_loss.max().item()]
    #             monitor_stats['hardness_range'] = [valid_hardness.min().item(), valid_hardness.max().item()]
    #             monitor_stats['scene_hardness_range'] = [valid_scene_hardness.min().item(),
    #                                                     valid_scene_hardness.max().item()]

    #             if global_hardness is not None:
    #                 monitor_stats['global_hardness_range'] = [valid_global_hardness.min().item(),
    #                                                         valid_global_hardness.max().item()]

    #             # 1. 前1%困难度对应的Loss均值对比
    #             if n_total >= 100:
    #                 k_1percent = max(1, n_total // 100)

    #                 # 获取前1%的索引
    #                 _, top1_hardness_idx = torch.topk(valid_hardness, k_1percent)
    #                 _, top1_scene_idx = torch.topk(valid_scene_hardness, k_1percent)

    #                 if global_hardness is not None:
    #                     _, top1_global_idx = torch.topk(valid_global_hardness, k_1percent)

    #                 # 计算对应的loss均值
    #                 top1_hardness_loss_mean = valid_loss[top1_hardness_idx].mean().item()
    #                 top1_scene_loss_mean = valid_loss[top1_scene_idx].mean().item()
    #                 overall_loss_mean = valid_loss.mean().item()

    #                 # 计算相对提升
    #                 hardness_relative_gain = (top1_hardness_loss_mean - overall_loss_mean) / overall_loss_mean * 100
    #                 scene_relative_gain = (top1_scene_loss_mean - overall_loss_mean) / overall_loss_mean * 100

    #                 monitor_stats['top1_hardness_loss_mean'] = top1_hardness_loss_mean
    #                 monitor_stats['top1_scene_loss_mean'] = top1_scene_loss_mean
    #                 monitor_stats['hardness_relative_gain'] = hardness_relative_gain
    #                 monitor_stats['scene_relative_gain'] = scene_relative_gain

    #                 if global_hardness is not None:
    #                     top1_global_loss_mean = valid_loss[top1_global_idx].mean().item()
    #                     global_relative_gain = (top1_global_loss_mean - overall_loss_mean) / overall_loss_mean * 100
    #                     monitor_stats['top1_global_loss_mean'] = top1_global_loss_mean
    #                     monitor_stats['global_relative_gain'] = global_relative_gain

    #                 # ==================== 新增：计算top1 precision指标 ====================
    #                 # 获取真实loss前1%的索引
    #                 _, top1_loss_idx = torch.topk(valid_loss, k_1percent)

    #                 # 转换为集合
    #                 top1_hardness_set = set(top1_hardness_idx.cpu().numpy())
    #                 top1_scene_set = set(top1_scene_idx.cpu().numpy())
    #                 top1_loss_set = set(top1_loss_idx.cpu().numpy())

    #                 # 计算交集
    #                 inter_hardness = len(top1_hardness_set & top1_loss_set)
    #                 inter_scene = len(top1_scene_set & top1_loss_set)

    #                 # 计算精确率和IoU
    #                 hardness_precision = inter_hardness / k_1percent
    #                 scene_precision = inter_scene / k_1percent

    #                 union_hardness = len(top1_hardness_set | top1_loss_set)
    #                 union_scene = len(top1_scene_set | top1_loss_set)

    #                 hardness_iou = inter_hardness / union_hardness if union_hardness > 0 else 0
    #                 scene_iou = inter_scene / union_scene if union_scene > 0 else 0

    #                 monitor_stats['hardness_precision'] = hardness_precision
    #                 monitor_stats['scene_precision'] = scene_precision
    #                 monitor_stats['hardness_iou'] = hardness_iou
    #                 monitor_stats['scene_iou'] = scene_iou

    #                 if global_hardness is not None:
    #                     top1_global_set = set(top1_global_idx.cpu().numpy())
    #                     inter_global = len(top1_global_set & top1_loss_set)
    #                     global_precision = inter_global / k_1percent
    #                     union_global = len(top1_global_set | top1_loss_set)
    #                     global_iou = inter_global / union_global if union_global > 0 else 0

    #                     monitor_stats['global_precision'] = global_precision
    #                     monitor_stats['global_iou'] = global_iou

    #             # 2. 相关性计算
    #             if len(valid_hardness) >= 2:
    #                 # 皮尔逊相关性
    #                 hardness_corr_matrix = torch.corrcoef(torch.stack([valid_hardness, valid_loss]))
    #                 hardness_corr = hardness_corr_matrix[0, 1].item()
    #                 monitor_stats['hardness_pearson_corr'] = hardness_corr

    #                 scene_corr_matrix = torch.corrcoef(torch.stack([valid_scene_hardness, valid_loss]))
    #                 scene_corr = scene_corr_matrix[0, 1].item()
    #                 monitor_stats['scene_pearson_corr'] = scene_corr

    #                 if global_hardness is not None:
    #                     global_corr_matrix = torch.corrcoef(torch.stack([valid_global_hardness, valid_loss]))
    #                     global_corr = global_corr_matrix[0, 1].item()
    #                     monitor_stats['global_pearson_corr'] = global_corr

    #                 # 斯皮尔曼相关性
    #                 def spearman_correlation_efficient_direct(x, y):
    #                     n = x.shape[0]
    #                     if n < 2:
    #                         return 0.0
    #                     x_ranks = torch.argsort(torch.argsort(x))
    #                     y_ranks = torch.argsort(torch.argsort(y))
    #                     corr_matrix = torch.corrcoef(torch.stack([x_ranks.float(), y_ranks.float()]))
    #                     return corr_matrix[0, 1].item()

    #                 monitor_stats['hardness_spearman_corr'] = spearman_correlation_efficient_direct(valid_hardness,
    #                                                                                                 valid_loss)
    #                 monitor_stats['scene_spearman_corr'] = spearman_correlation_efficient_direct(
    #                     valid_scene_hardness, valid_loss)
    #                 if global_hardness is not None:
    #                     monitor_stats['global_spearman_corr'] = spearman_correlation_efficient_direct(
    #                         valid_global_hardness, valid_loss)

    #     # ==================== 返回所有信号 ====================
    #     res = {
    #         'output_voxels': output['occ'],
    #         'output_voxels_fine': output.get('fine_output', None),
    #         'output_coords_fine': output.get('fine_coord', None),
    #         'hardness_pred': hardness_pred,  # 体素级困难度 [B, H, W, D]
    #         'scene_hardness_pred': scene_hardness_pred,  # 场景级困难度 [B, H, W, D]
    #         'global_hardness': global_hardness,  # 全局困难度 [B, H, W, D]
    #         'loss_distribution': loss_distribution,  # 损失分布 [B, H, W, D]
    #         'monitor_stats': monitor_stats,  # 原有统计信号
    #         'visible_monitor_stats': visible_monitor_stats,  # 新增：visible区域统计信号
    #     }

    #     return res






    @force_fp32()
    def forward_train(self, voxel_feats, img_feats=None, pts_feats=None, transform=None, gt_occupancy=None, gt_occupancy_flow=None, **kwargs):
        mark = kwargs.get('mark', None)
        if mark:
            with torch.no_grad():
                res = self.forward(voxel_feats, img_feats=img_feats, pts_feats=pts_feats, transform=transform, **kwargs)
            results = kwargs.get('results', None)
            loss = self.enhanced_loss(output_voxels=res['output_voxels'], hardness_pred=results['bev_hardness'], scene_hardness_pred=results['scene_hardness'],
                                      target_voxels=gt_occupancy)

        else:
            res = self.forward(voxel_feats, img_feats=img_feats, pts_feats=pts_feats, transform=transform, **kwargs)
            loss = self.loss(target_voxels=gt_occupancy,
                output_voxels = res['output_voxels'],
                output_coords_fine=res['output_coords_fine'],
                output_voxels_fine=res['output_voxels_fine'])

        return loss


    @force_fp32() 
    def loss_voxel(self, output_voxels, target_voxels, tag):

        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        # output_voxels = torch.log(output_voxels * 0) + output_voxels/0 # debug !!!!!!!!

        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        if self.use_focal_loss:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        else:
            loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
        loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
        loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        if self.use_dice_loss:
            visible_mask = target_voxels!=255
            visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
            visible_target_voxels = target_voxels[visible_mask]
            visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
            loss_dict['loss_voxel_dice_{}'.format(tag)] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict

    @force_fp32() 
    def loss(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None, visible_mask=None, **kwargs):
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='refined_{}'.format(index)))
        return loss_dict

    @force_fp32()
    def enhanced_loss(self, output_voxels=None, hardness_pred=None, scene_hardness_pred=None, target_voxels=None, **kwargs):
        """
        增强的loss函数，包含HPNet监督
        """
        loss_dict = {}

        # # 1. 原有的occupancy loss
        # for index, output_voxel in enumerate(output_voxels):
        #     loss_dict.update(self.loss_voxel(output_voxel, target_voxels, tag='c_{}'.format(index)))

        # 2. HPNet监督loss
        if hardness_pred is not None and len(output_voxels) > 0:
            # ==================== 基础计算 ====================
            loss_calculator = LossDistributionCalculator(
                use_focal_loss=self.use_focal_loss,
                class_weights=self.class_weights,
                empty_idx=self.empty_idx
            )

            with torch.no_grad():
                true_loss_dist, valid_mask = loss_calculator(output_voxels[0].detach(), target_voxels)

            B, H_true, W_true, D_true = true_loss_dist.shape

            # 上采样hardness预测
            hardness_pred_upsampled = F.interpolate(
                hardness_pred.unsqueeze(1),
                size=(H_true, W_true, D_true),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)

            scene_hardness_pred_upsampled = F.interpolate(
                scene_hardness_pred.unsqueeze(1),
                size=(H_true, W_true, D_true),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)

            # ==================== 计算HPNet监督loss ====================
            hp_supervision_loss = HPNSupervisionLoss(
                loss_type='top1_percent_focus',
                temperature=0.1,
                weight=5.0
            )

            loss_hp = hp_supervision_loss(hardness_pred_upsampled, true_loss_dist, valid_mask)
            loss_dict['loss_hp_supervision'] = loss_hp

            loss_scene_hp = hp_supervision_loss(scene_hardness_pred_upsampled, true_loss_dist, valid_mask)
            loss_dict['loss_scene_hp_supervision'] = loss_scene_hp

            # ==================== 监控与统计（仅cuda:0） ====================
            if str(hardness_pred_upsampled.device) == 'cuda:0':
                # 准备监控数据
                valid_scene_hardness = scene_hardness_pred_upsampled[valid_mask].flatten()
                valid_hardness = hardness_pred_upsampled[valid_mask].flatten()
                valid_loss = true_loss_dist[valid_mask].flatten()
                
                # 计算全局困难度（基于置信度）
                with torch.no_grad():
                    logits = output_voxels[0].detach()
                    logits_excluding_ignore = logits[:, 1:, :, :, :]
                    probs = F.softmax(logits_excluding_ignore, dim=1)
                    top2_probs, _ = torch.topk(probs, k=2, dim=1)
                    prob_diff = top2_probs[:, 0, ...] - top2_probs[:, 1, ...]
                    epsilon = 1e-3
                    prob_diff = torch.clamp(prob_diff, min=epsilon, max=1.0-epsilon)
                    global_hardness_raw = 1.0 / prob_diff
                    global_hardness = torch.log(global_hardness_raw)
                    valid_global_hardness = global_hardness[valid_mask].flatten()
                
                # ==================== 精简的对比统计 ====================
                n_total = len(valid_hardness)
                
                # 计算总体平均loss
                overall_loss_mean = valid_loss.mean().item()
                
                # 0. 前1%困难度对应的loss均值对比（新增）
                if n_total >= 100:
                    k_1percent = max(1, n_total // 100)
                    
                    # 获取前1%的索引
                    _, top1_hardness_idx = torch.topk(valid_hardness, k_1percent)
                    _, top1_scene_idx = torch.topk(valid_scene_hardness, k_1percent)
                    _, top1_global_idx = torch.topk(valid_global_hardness, k_1percent)
                    
                    # 计算对应的loss均值
                    top1_hardness_loss_mean = valid_loss[top1_hardness_idx].mean().item()
                    top1_scene_loss_mean = valid_loss[top1_scene_idx].mean().item()
                    top1_global_loss_mean = valid_loss[top1_global_idx].mean().item()
                    
                    # 计算相对提升（相对于总体平均）
                    hardness_relative_gain = (top1_hardness_loss_mean - overall_loss_mean) / overall_loss_mean * 100
                    scene_relative_gain = (top1_scene_loss_mean - overall_loss_mean) / overall_loss_mean * 100
                    global_relative_gain = (top1_global_loss_mean - overall_loss_mean) / overall_loss_mean * 100
                    
                    # 计算相对于随机选择的提升倍数
                    # 随机选择k个样本的期望loss均值就是总体平均loss
                    hardness_gain_ratio = top1_hardness_loss_mean / overall_loss_mean
                    scene_gain_ratio = top1_scene_loss_mean / overall_loss_mean
                    global_gain_ratio = top1_global_loss_mean / overall_loss_mean
                    
                    print("\n" + "="*60)
                    print("前1%困难度对应的Loss均值对比（关键指标）")
                    print("="*60)
                    print(f"总体平均Loss: {overall_loss_mean:.4f}")
                    print(f"{'类型':<10} {'前1% Loss均值':<15} {'相对提升':<15} {'提升倍数':<15}")
                    print(f"{'体素级':<10} {top1_hardness_loss_mean:<15.4f} {hardness_relative_gain:<15.1f}% {hardness_gain_ratio:<15.2f}x")
                    print(f"{'场景级':<10} {top1_scene_loss_mean:<15.4f} {scene_relative_gain:<15.1f}% {scene_gain_ratio:<15.2f}x")
                    print(f"{'全局困难度':<10} {top1_global_loss_mean:<15.4f} {global_relative_gain:<15.1f}% {global_gain_ratio:<15.2f}x")
                    
                    # 计算识别效率（前1%中loss在前1%的比例 = 精确率）
                    # 这部分已经在后面的统计中有了，但这里可以再强调一下
                
                # 1. 相关性对比表格
                print("\n" + "="*60)
                print("三种困难度相关性对比")
                print("="*60)
                
                # 全部有效点相关性
                if len(valid_hardness) >= 2:
                    # 体素级相关性
                    hardness_corr_matrix = torch.corrcoef(torch.stack([valid_hardness, valid_loss]))
                    hardness_corr = hardness_corr_matrix[0, 1].item()
                    hardness_spearman = self.spearman_correlation_efficient_direct(valid_hardness, valid_loss)
                    
                    # 场景级相关性
                    scene_corr_matrix = torch.corrcoef(torch.stack([valid_scene_hardness, valid_loss]))
                    scene_corr = scene_corr_matrix[0, 1].item()
                    scene_spearman = self.spearman_correlation_efficient_direct(valid_scene_hardness, valid_loss)
                    
                    # 全局困难度相关性
                    global_corr_matrix = torch.corrcoef(torch.stack([valid_global_hardness, valid_loss]))
                    global_corr = global_corr_matrix[0, 1].item()
                    global_spearman = self.spearman_correlation_efficient_direct(valid_global_hardness, valid_loss)
                    
                    print(f"{'类型':<10} {'皮尔逊相关性':<15} {'斯皮尔曼相关性':<15}")
                    print(f"{'体素级':<10} {hardness_corr:<15.3f} {hardness_spearman:<15.3f}")
                    print(f"{'场景级':<10} {scene_corr:<15.3f} {scene_spearman:<15.3f}")
                    print(f"{'全局困难度':<10} {global_corr:<15.3f} {global_spearman:<15.3f}")
                
                # 2. 前1%重叠统计对比
                if n_total >= 100:
                    k_1percent = max(1, n_total // 100)
                    
                    # 获取前1%的索引（这里重新获取一遍，避免重复计算）
                    _, top1_hardness_idx = torch.topk(valid_hardness, k_1percent)
                    _, top1_scene_idx = torch.topk(valid_scene_hardness, k_1percent)
                    _, top1_global_idx = torch.topk(valid_global_hardness, k_1percent)
                    _, top1_loss_idx = torch.topk(valid_loss, k_1percent)
                    
                    # 转换为集合
                    top1_hardness_set = set(top1_hardness_idx.cpu().numpy())
                    top1_scene_set = set(top1_scene_idx.cpu().numpy())
                    top1_global_set = set(top1_global_idx.cpu().numpy())
                    top1_loss_set = set(top1_loss_idx.cpu().numpy())
                    
                    # 计算交集
                    inter_hardness = len(top1_hardness_set & top1_loss_set)
                    inter_scene = len(top1_scene_set & top1_loss_set)
                    inter_global = len(top1_global_set & top1_loss_set)
                    
                    # 计算精确率和IoU
                    precision_hardness = inter_hardness / k_1percent
                    precision_scene = inter_scene / k_1percent
                    precision_global = inter_global / k_1percent
                    
                    union_hardness = len(top1_hardness_set | top1_loss_set)
                    union_scene = len(top1_scene_set | top1_loss_set)
                    union_global = len(top1_global_set | top1_loss_set)
                    
                    iou_hardness = inter_hardness / union_hardness if union_hardness > 0 else 0
                    iou_scene = inter_scene / union_scene if union_scene > 0 else 0
                    iou_global = inter_global / union_global if union_global > 0 else 0
                    
                    print("\n" + "="*60)
                    print("前1%重叠统计对比")
                    print("="*60)
                    print(f"{'类型':<10} {'交集数量':<10} {'精确率':<10} {'IoU':<10}")
                    print(f"{'体素级':<10} {inter_hardness:<10} {precision_hardness:<10.3f} {iou_hardness:<10.3f}")
                    print(f"{'场景级':<10} {inter_scene:<10} {precision_scene:<10.3f} {iou_scene:<10.3f}")
                    print(f"{'全局困难度':<10} {inter_global:<10} {precision_global:<10.3f} {iou_global:<10.3f}")
                
                # 3. 前1%困难度 vs 前5%真实loss对比
                if n_total >= 100:
                    k_5percent_loss = max(1, n_total // 20)
                    k_1percent = max(1, n_total // 100)
                    
                    # 获取前5%loss的索引
                    _, top5_loss_idx = torch.topk(valid_loss, k_5percent_loss)
                    top5_loss_set = set(top5_loss_idx.cpu().numpy())
                    
                    # 计算与各种困难度前1%的交集
                    inter_hardness_5loss = len(top1_hardness_set & top5_loss_set)
                    inter_scene_5loss = len(top1_scene_set & top5_loss_set)
                    inter_global_5loss = len(top1_global_set & top5_loss_set)
                    
                    # 计算精确率、召回率、F1、IoU
                    precision_hardness_5loss = inter_hardness_5loss / k_1percent
                    precision_scene_5loss = inter_scene_5loss / k_1percent
                    precision_global_5loss = inter_global_5loss / k_1percent
                    
                    recall_hardness_5loss = inter_hardness_5loss / k_5percent_loss
                    recall_scene_5loss = inter_scene_5loss / k_5percent_loss
                    recall_global_5loss = inter_global_5loss / k_5percent_loss
                    
                    # 计算F1分数
                    def compute_f1(precision, recall):
                        if precision + recall > 0:
                            return 2 * precision * recall / (precision + recall)
                        return 0.0
                    
                    f1_hardness = compute_f1(precision_hardness_5loss, recall_hardness_5loss)
                    f1_scene = compute_f1(precision_scene_5loss, recall_scene_5loss)
                    f1_global = compute_f1(precision_global_5loss, recall_global_5loss)
                    
                    # 计算IoU
                    union_hardness_5loss = len(top1_hardness_set | top5_loss_set)
                    union_scene_5loss = len(top1_scene_set | top5_loss_set)
                    union_global_5loss = len(top1_global_set | top5_loss_set)
                    
                    iou_hardness_5loss = inter_hardness_5loss / union_hardness_5loss if union_hardness_5loss > 0 else 0
                    iou_scene_5loss = inter_scene_5loss / union_scene_5loss if union_scene_5loss > 0 else 0
                    iou_global_5loss = inter_global_5loss / union_global_5loss if union_global_5loss > 0 else 0
                    
                    print("\n" + "="*60)
                    print("前1%困难度 vs 前5%真实loss对比")
                    print("="*60)
                    print(f"{'类型':<10} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'IoU':<10}")
                    print(f"{'体素级':<10} {precision_hardness_5loss:<10.3f} {recall_hardness_5loss:<10.3f} {f1_hardness:<10.3f} {iou_hardness_5loss:<10.3f}")
                    print(f"{'场景级':<10} {precision_scene_5loss:<10.3f} {recall_scene_5loss:<10.3f} {f1_scene:<10.3f} {iou_scene_5loss:<10.3f}")
                    print(f"{'全局困难度':<10} {precision_global_5loss:<10.3f} {recall_global_5loss:<10.3f} {f1_global:<10.3f} {iou_global_5loss:<10.3f}")
                    
                    # 打印基本统计信息（简要）
                    print("\n" + "="*60)
                    print("基本统计信息")
                    print("="*60)
                    print(f"有效体素总数: {n_total}")
                    print(f"体素级困难度范围: [{valid_hardness.min():.3f}, {valid_hardness.max():.3f}]")
                    print(f"场景级困难度范围: [{valid_scene_hardness.min():.3f}, {valid_scene_hardness.max():.3f}]")
                    print(f"全局困难度范围: [{valid_global_hardness.min():.3f}, {valid_global_hardness.max():.3f}]")
                    print(f"真实loss范围: [{valid_loss.min():.3f}, {valid_loss.max():.3f}]")

            return loss_dict

    def spearman_correlation_efficient_direct(self, x, y):
        """直接计算两个一维张量的斯皮尔曼相关性"""
        n = x.shape[0]
        if n < 2:
            return 0.0

        x_ranks = torch.argsort(torch.argsort(x))
        y_ranks = torch.argsort(torch.argsort(y))

        corr_matrix = torch.corrcoef(torch.stack([x_ranks.float(), y_ranks.float()]))
        return corr_matrix[0, 1].item()

