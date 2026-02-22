import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from ..layer import DeformableTransformerLayer, TransformerLayer
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import HEADS
import torch.utils.checkpoint as cp


@HEADS.register_module()
class MultiScaleInstanceFusionModule(BaseModule):
    def __init__(self, embed_dims=80, num_global_instances=120,
                 num_heads=8, num_points=4,
                 # 各尺度的instance数量
                 scale1_instances_per_camera=25,  # 原尺寸 (16×44),30
                 scale2_instances_per_camera=10,  # 1/2尺寸 (8×22)
                 scale3_instances_per_camera=7,  # 1/4尺寸 (4×11)
                 use_multi_scale=True,
                 with_cp=True,
                 ):
        super(MultiScaleInstanceFusionModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_cameras = 6
        self.num_global_instances = num_global_instances
        self.with_cp = with_cp

        # 各尺度instance数量
        self.scale1_instances_per_camera = scale1_instances_per_camera
        self.scale2_instances_per_camera = scale2_instances_per_camera
        self.scale3_instances_per_camera = scale3_instances_per_camera
        self.use_multi_scale = use_multi_scale

        # 计算各尺度总instance数
        self.total_scale1_instances = self.num_cameras * scale1_instances_per_camera  # 180
        self.total_scale2_instances = self.num_cameras * scale2_instances_per_camera  # 60
        self.total_scale3_instances = self.num_cameras * scale3_instances_per_camera  # 30
        self.total_instances = self.total_scale1_instances + self.total_scale2_instances + self.total_scale3_instances  # 270

        # ========== 各尺度的实例组件 ==========
        # 尺度1: 原尺寸实例 - 并行化：每个尺度只用一个注意力层，但处理所有相机
        self.scale1_inst_embed = nn.Embedding(self.total_scale1_instances, embed_dims)
        self.scale1_pred_pts = nn.Embedding(self.total_scale1_instances, 2)
        self.scale1_deform_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels=self.num_cameras, num_points=num_points
        )  # num_levels设置为相机数，支持多相机并行

        # 尺度2: 1/2尺寸实例
        self.scale2_inst_embed = nn.Embedding(self.total_scale2_instances, embed_dims)
        self.scale2_pred_pts = nn.Embedding(self.total_scale2_instances, 2)
        self.scale2_deform_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels=self.num_cameras, num_points=num_points
        )

        # 尺度3: 1/4尺寸实例
        self.scale3_inst_embed = nn.Embedding(self.total_scale3_instances, embed_dims)
        self.scale3_pred_pts = nn.Embedding(self.total_scale3_instances, 2)
        self.scale3_deform_attn = DeformableTransformerLayer(
            embed_dims, num_heads, num_levels=self.num_cameras, num_points=num_points
        )

        # ========== 保留的全局组件 ==========
        self.global_inst_embed = nn.Embedding(num_global_instances, embed_dims)
        self.history_inst_embed = self.global_inst_embed  # 权重共享

        # 仅保留这两个Transformer层
        self.camera_to_global_cross_attn = TransformerLayer(embed_dims, num_heads)
        self.inst_to_hard_voxel_cross_attn = TransformerLayer(embed_dims, num_heads)

        # 添加用于历史实例融合的self-attention层
        self.global_hist_self_attn = TransformerLayer(embed_dims, num_heads)

        # 初始化参考点
        self._init_scale_reference_points()

    def process_single_scale_parallel(self, context, scale_idx):
        """并行处理单尺度下所有相机的实例"""
        bs = context.shape[0]

        if scale_idx == 1:
            inst_embed = self.scale1_inst_embed
            pred_pts = self.scale1_pred_pts
            deform_attn = self.scale1_deform_attn
            instances_per_cam = self.scale1_instances_per_camera
            scale_factor = 1
        elif scale_idx == 2:
            inst_embed = self.scale2_inst_embed
            pred_pts = self.scale2_pred_pts
            deform_attn = self.scale2_deform_attn
            instances_per_cam = self.scale2_instances_per_camera
            scale_factor = 2
        else:  # scale_idx == 3
            inst_embed = self.scale3_inst_embed
            pred_pts = self.scale3_pred_pts
            deform_attn = self.scale3_deform_attn
            instances_per_cam = self.scale3_instances_per_camera
            scale_factor = 4

        # ===== 并行化关键：一次性准备所有相机的查询和参考点 =====
        # 所有实例查询 [B, total_instances, C]
        queries = inst_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # 所有参考点 [B, total_instances, 2]
        ref_pts = pred_pts.weight.unsqueeze(0).repeat(bs, 1, 1).sigmoid()

        # ===== 并行化关键：一次性处理所有相机的特征 =====
        B, N, C, H, W = context.shape  # N = num_cameras

        # 根据尺度进行下采样
        if scale_factor > 1:
            # 一次性对所有相机进行下采样
            context_reshaped = context.view(B * N, C, H, W)
            H_scaled, W_scaled = H // scale_factor, W // scale_factor
            scaled_feat = F.interpolate(
                context_reshaped,
                size=(H_scaled, W_scaled),
                mode='bilinear',
                align_corners=False
            )
            scaled_feat = scaled_feat.view(B, N, C, H_scaled, W_scaled)
        else:
            scaled_feat = context
            H_scaled, W_scaled = H, W

        # ===== 并行化关键：展平所有相机的特征 =====
        # [B, N, C, H_scaled, W_scaled] -> [B, N*H_scaled*W_scaled, C]
        feat_flatten = scaled_feat.view(B, N, C, -1).permute(0, 1, 3, 2).reshape(B, -1, C)

        # ===== 准备空间形状和层级索引 =====
        # 每个相机有相同的空间形状
        spatial_shapes = torch.tensor(
            [[H_scaled, W_scaled]],
            dtype=torch.long,
            device=context.device
        ).repeat(N, 1)

        # 层级起始索引：每个相机特征在展平后的起始位置
        level_start_index = torch.tensor(
            [i * H_scaled * W_scaled for i in range(N)],
            dtype=torch.long,
            device=context.device
        )

        # ===== 单次注意力计算（批处理所有相机） =====
        if self.with_cp and torch.is_grad_enabled():
            output_queries = cp.checkpoint(
                deform_attn,
                queries,
                feat_flatten,
                None,
                ref_pts.unsqueeze(2),  # [B, total_instances, 1, 2]
                spatial_shapes,
                level_start_index
            )
        else:
            output_queries = deform_attn(
                queries,
                feat_flatten,
                query_pos=None,
                ref_pts=ref_pts.unsqueeze(2),
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index
            )

        return output_queries

    def process_multi_scale_instances(self, context):
        """使用CUDA流并行处理所有尺度"""
        if not self.use_multi_scale:
            return self.process_single_scale_parallel(context, 1)

        # 创建三个CUDA流
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()
        stream3 = torch.cuda.Stream()

        # 在stream1中处理尺度1
        with torch.cuda.stream(stream1):
            scale1_instances = self.process_single_scale_parallel(context, 1)

        # 在stream2中处理尺度2
        with torch.cuda.stream(stream2):
            scale2_instances = self.process_single_scale_parallel(context, 2)

        # 在stream3中处理尺度3
        with torch.cuda.stream(stream3):
            scale3_instances = self.process_single_scale_parallel(context, 3)

        # 同步所有流
        torch.cuda.synchronize()

        # 合并所有尺度的实例
        all_instances = torch.cat([scale1_instances, scale2_instances, scale3_instances], dim=1)
        return all_instances

    def fuse_instances_to_global(self, instances):
        """简化的实例到全局实例融合"""
        bs = instances.shape[0]

        # 初始化全局实例查询
        global_inst_queries = self.global_inst_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        # 实例到全局实例的交叉注意力
        if self.with_cp and torch.is_grad_enabled():
            global_inst_queries = cp.checkpoint(
                self.camera_to_global_cross_attn,
                global_inst_queries, instances, instances
            )
        else:
            global_inst_queries = self.camera_to_global_cross_attn(
                global_inst_queries, instances, instances)

        return global_inst_queries

    def fuse_history_with_global_instances(self, global_inst_queries, history_inst_queries):
        """历史实例融合 - 使用拼接+self-attn的方式"""
        bs = global_inst_queries.shape[0]

        # 拼接当前实例和历史实例 [B, 2*num_global_instances, C]
        combined_instances = torch.cat([global_inst_queries, history_inst_queries], dim=1)

        # 使用self-attention进行信息融合
        if self.with_cp and torch.is_grad_enabled():
            fused_combined = cp.checkpoint(self.global_hist_self_attn, combined_instances)
        else:
            fused_combined = self.global_hist_self_attn(combined_instances)

        # 拆分回当前实例和历史实例
        fused_global_inst = fused_combined[:, :self.num_global_instances, :]
        updated_history_inst = fused_combined[:, self.num_global_instances:, :]

        return fused_global_inst, updated_history_inst

    def forward(self, context, hard_voxel_feat, history_inst_queries):
        """改进版本 - 三尺度独立提取实例，但每个尺度内多相机并行"""
        bs = context.shape[0]

        # 步骤1: 多尺度实例提取（现在每个尺度内部是并行的）
        multi_scale_instances = self.process_multi_scale_instances(context)

        # 步骤2: 融合实例为全局实例
        global_inst_queries = self.fuse_instances_to_global(multi_scale_instances)

        # 步骤3: 全局实例与历史实例融合（使用新的拼接+self-attn方式）
        fused_global_inst, updated_history_inst = self.fuse_history_with_global_instances(
            global_inst_queries, history_inst_queries)

        # 步骤4: 与困难体素交互
        if hard_voxel_feat is not None and hard_voxel_feat.shape[1] > 0:
            # 使用融合后的全局实例与历史实例共同增强困难体素
            all_instances = torch.cat([fused_global_inst, updated_history_inst], dim=1)

            if self.with_cp and torch.is_grad_enabled():
                enhanced_hard_voxel = cp.checkpoint(
                    self.inst_to_hard_voxel_cross_attn,
                    hard_voxel_feat, all_instances, all_instances
                )
            else:
                enhanced_hard_voxel = self.inst_to_hard_voxel_cross_attn(
                    hard_voxel_feat, all_instances, all_instances
                )
        else:
            enhanced_hard_voxel = hard_voxel_feat

        return fused_global_inst, updated_history_inst, enhanced_hard_voxel, multi_scale_instances

    def _init_scale_reference_points(self):
        """初始化各尺度的参考点"""
        with torch.no_grad():
            # 尺度1参考点（密集分布）
            self._init_single_scale_points(
                self.scale1_pred_pts, self.scale1_instances_per_camera, 0.1, 0.9)

            # 尺度2参考点（中等分布）
            self._init_single_scale_points(
                self.scale2_pred_pts, self.scale2_instances_per_camera, 0.2, 0.8)

            # 尺度3参考点（稀疏分布）
            self._init_single_scale_points(
                self.scale3_pred_pts, self.scale3_instances_per_camera, 0.3, 0.7)

    def _init_single_scale_points(self, pred_pts_layer, instances_per_camera, low, high):
        """初始化单个尺度的参考点"""
        total_instances = self.num_cameras * instances_per_camera

        grid_size = int(math.sqrt(instances_per_camera))
        if grid_size * grid_size < instances_per_camera:
            grid_size += 1

        # 根据尺度调整分布范围
        x_coords = torch.linspace(low, high, grid_size)
        y_coords = torch.linspace(low, high, grid_size)

        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        if grid_points.shape[0] > instances_per_camera:
            indices = torch.randperm(grid_points.shape[0])[:instances_per_camera]
            base_points = grid_points[indices]
        else:
            base_points = grid_points
            while base_points.shape[0] < instances_per_camera:
                remaining = instances_per_camera - base_points.shape[0]
                additional = grid_points[:min(remaining, grid_points.shape[0])]
                base_points = torch.cat([base_points, additional], dim=0)

        all_points = []
        for cam_idx in range(self.num_cameras):
            if cam_idx == 0:
                cam_points = base_points.clone()
            else:
                noise = torch.randn_like(base_points) * 0.01
                cam_points = base_points + noise
                cam_points = torch.clamp(cam_points, low, high)
            all_points.append(cam_points)

        all_points = torch.cat(all_points, dim=0)
        logit_points = torch.log(all_points / (1 - all_points))
        pred_pts_layer.weight.data = logit_points