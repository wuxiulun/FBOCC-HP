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
            # 计算真实的loss分布和有效掩码
            loss_calculator = LossDistributionCalculator(
                use_focal_loss=self.use_focal_loss,
                class_weights=self.class_weights,
                empty_idx=self.empty_idx
            )

            # 计算loss分布和有效掩码
            with torch.no_grad():
                true_loss_dist, valid_mask = loss_calculator(output_voxels[0].detach(), target_voxels)

            # 将hardness_pred上采样到真实loss分布的分辨率
            B, H_true, W_true, D_true = true_loss_dist.shape  # [B, 200, 200, 16]

            # 使用三线性插值上采样
            hardness_pred_upsampled = F.interpolate(
                hardness_pred.unsqueeze(1),  # 添加通道维度 [B, 1, 100, 100, 8]
                size=(H_true, W_true, D_true),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)  # [B, 200, 200, 16]

            scene_hardness_pred_upsampled = F.interpolate(
                scene_hardness_pred.unsqueeze(1),  # 添加通道维度 [B, 1, 100, 100, 8]
                size=(H_true, W_true, D_true),
                mode='trilinear',
                align_corners=False
            ).squeeze(1)  # [B, 200, 200, 16]

            # 计算HPNet监督loss（传入有效掩码）
            hp_supervision_loss = HPNSupervisionLoss(
                loss_type='top1_percent_focus',
                temperature=0.1,
                weight=5.0
            )

            loss_hp = hp_supervision_loss(hardness_pred_upsampled, true_loss_dist, valid_mask)
            loss_dict['loss_hp_supervision'] = loss_hp

            loss_scene_hp = hp_supervision_loss(scene_hardness_pred_upsampled, true_loss_dist, valid_mask)
            loss_dict['loss_scene_hp_supervision'] = loss_scene_hp

            if str(hardness_pred_upsampled.device) == 'cuda:0':
                # 监控信息（使用上采样后的hardness）
                valid_scene_hardness = scene_hardness_pred_upsampled[valid_mask].flatten()
                valid_hardness = hardness_pred_upsampled[valid_mask].flatten()
                valid_loss = true_loss_dist[valid_mask].flatten()

                print(f"scnen Hardness range: [{valid_scene_hardness.min():.3f}, {valid_scene_hardness.max():.3f}]")
                print(f"scene Hardness mean: {valid_scene_hardness.mean():.3f}")
                print(f"Hardness range: [{valid_hardness.min():.3f}, {valid_hardness.max():.3f}]")
                print(f"Hardness mean: {valid_hardness.mean():.3f}")
                print(f"Loss range: [{valid_loss.min():.3f}, {valid_loss.max():.3f}]")
                print(f"Loss mean: {valid_loss.mean():.3f}")

                # 统计前N名
                if len(valid_hardness) > 3000:
                    scene_hardness_sorted, _ = torch.sort(valid_scene_hardness, descending=True)
                    hardness_sorted, _ = torch.sort(valid_hardness, descending=True)
                    loss_sorted, _ = torch.sort(valid_loss, descending=True)
                    print(f"scene Hardness 第3000名: {scene_hardness_sorted[2999]:.3f}")
                    print(f"Hardness 第3000名: {hardness_sorted[2999]:.3f}")
                    print(f"Loss 第3000名: {loss_sorted[2999]:.3f}")
                else:
                    print(f"有效区域不足3000个: {len(valid_hardness)}")

                # 使用所有有效点计算相关性
                if len(valid_hardness) >= 2:
                    # 皮尔逊相关性
                    correlation_matrix = torch.corrcoef(torch.stack([valid_hardness, valid_loss]))
                    correlation = correlation_matrix[0, 1].item()

                    # 斯皮尔曼相关性
                    spearman_corr = self.spearman_correlation_efficient_direct(valid_hardness, valid_loss)

                    print(f"皮尔逊相关性(全部{len(valid_hardness)}个有效点): {correlation:.3f}")
                    print(f"斯皮尔曼相关性(全部{len(valid_hardness)}个有效点): {spearman_corr:.3f}")

                    # 额外统计：高loss区域的相关性（前1%）
                    if len(valid_hardness) >= 10:
                        k = max(1000, len(valid_hardness) // 100)  # 至少1000个点，最多前1%

                        # 按loss值排序，取前k个
                        topk_loss, topk_indices = torch.topk(valid_loss, k)
                        topk_hardness = valid_hardness[topk_indices]

                        # 计算高loss区域的相关性
                        high_loss_correlation_matrix = torch.corrcoef(torch.stack([topk_hardness, topk_loss]))
                        high_loss_correlation = high_loss_correlation_matrix[0, 1].item()

                        high_loss_spearman = self.spearman_correlation_efficient_direct(topk_hardness, topk_loss)

                        print(f"高loss区域皮尔逊相关性(前{k}个): {high_loss_correlation:.3f}")
                        print(f"高loss区域斯皮尔曼相关性(前{k}个): {high_loss_spearman:.3f}")

                if len(valid_hardness) >= 100:  # 至少100个点才能计算1%
                    n_total = len(valid_hardness)
                    k_1percent = max(1, n_total // 100)  # 前1%的数量

                    # 获取前1%预测困难度的索引
                    _, top1_hardness_indices = torch.topk(valid_hardness, k_1percent)

                    # 获取前1%真实loss的索引
                    _, top1_loss_indices = torch.topk(valid_loss, k_1percent)

                    # 转换为集合以便计算交集
                    top1_hardness_set = set(top1_hardness_indices.cpu().numpy())
                    top1_loss_set = set(top1_loss_indices.cpu().numpy())

                    # 计算交集
                    intersection_set = top1_hardness_set & top1_loss_set
                    n_intersection = len(intersection_set)

                    # 计算各种指标
                    # 1. 精确率：预测为困难的体素中，真正是困难的比例
                    precision = n_intersection / k_1percent if k_1percent > 0 else 0

                    # 4. Jaccard相似度（IoU）
                    union_set = top1_hardness_set | top1_loss_set
                    jaccard_similarity = n_intersection / len(union_set) if len(union_set) > 0 else 0

                    print(f"=== 前1%重叠统计 ===")
                    print(f"交集数量: {n_intersection}")
                    print(f"精确率(预测前1%中真实前1%的比例): {precision:.3f}")
                    print(f"Jaccard相似度(IoU): {jaccard_similarity:.3f}")




                # 使用所有有效点计算相关性
                if len(valid_scene_hardness) >= 2:
                    # 皮尔逊相关性
                    correlation_matrix = torch.corrcoef(torch.stack([valid_scene_hardness, valid_loss]))
                    correlation = correlation_matrix[0, 1].item()

                    # 斯皮尔曼相关性
                    spearman_corr = self.spearman_correlation_efficient_direct(valid_scene_hardness, valid_loss)

                    print(f"场景级皮尔逊相关性(全部{len(valid_scene_hardness)}个有效点): {correlation:.3f}")
                    print(f"场景级斯皮尔曼相关性(全部{len(valid_scene_hardness)}个有效点): {spearman_corr:.3f}")

                    # 额外统计：高loss区域的相关性（前1%）
                    if len(valid_scene_hardness) >= 10:
                        k = max(1000, len(valid_scene_hardness) // 100)  # 至少1000个点，最多前1%

                        # 按loss值排序，取前k个
                        topk_loss, topk_indices = torch.topk(valid_loss, k)
                        topk_scene_hardness = valid_scene_hardness[topk_indices]

                        # 计算高loss区域的相关性
                        high_loss_correlation_matrix = torch.corrcoef(torch.stack([topk_scene_hardness, topk_loss]))
                        high_loss_correlation = high_loss_correlation_matrix[0, 1].item()

                        high_loss_spearman = self.spearman_correlation_efficient_direct(topk_scene_hardness, topk_loss)

                        print(f"场景级高loss区域皮尔逊相关性(前{k}个): {high_loss_correlation:.3f}")
                        print(f"场景级高loss区域斯皮尔曼相关性(前{k}个): {high_loss_spearman:.3f}")

                if len(valid_scene_hardness) >= 100:  # 至少100个点才能计算1%
                    n_total = len(valid_scene_hardness)
                    k_1percent = max(1, n_total // 100)  # 前1%的数量

                    # 获取前1%预测困难度的索引
                    _, top1_scene_hardness_indices = torch.topk(valid_scene_hardness, k_1percent)

                    # 获取前1%真实loss的索引
                    _, top1_loss_indices = torch.topk(valid_loss, k_1percent)

                    # 转换为集合以便计算交集
                    top1_scene_hardness_set = set(top1_scene_hardness_indices.cpu().numpy())
                    top1_loss_set = set(top1_loss_indices.cpu().numpy())

                    # 计算交集
                    intersection_set = top1_scene_hardness_set & top1_loss_set
                    n_intersection = len(intersection_set)

                    # 计算各种指标
                    # 1. 精确率：预测为困难的体素中，真正是困难的比例
                    precision = n_intersection / k_1percent if k_1percent > 0 else 0

                    # 4. Jaccard相似度（IoU）
                    union_set = top1_scene_hardness_set | top1_loss_set
                    jaccard_similarity = n_intersection / len(union_set) if len(union_set) > 0 else 0

                    print(f"=== 前1%场景级重叠统计 ===")
                    print(f"场景级交集数量: {n_intersection}")
                    print(f"场景级精确率(预测前1%中真实前1%的比例): {precision:.3f}")
                    print(f"场景级Jaccard相似度(IoU): {jaccard_similarity:.3f}")
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

