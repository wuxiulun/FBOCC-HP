import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_voxel_wise_focal_loss_fbocc_style(pred, target, class_weights, gamma=2.0, ignore_index=255):
    """
    按照FBOCC风格实现Focal Loss（使用class_weights作为alpha）
    """
    B, C, H, W, D = pred.shape

    # 确保class_weights与pred在同一个设备和数据类型上
    class_weights = class_weights.to(pred.device).to(pred.dtype)

    # 将pred和target展平
    pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
    target_flat = target.reshape(-1)

    # 过滤ignore区域
    valid_mask = (target_flat != ignore_index)
    valid_pred = pred_flat[valid_mask]
    valid_target = target_flat[valid_mask]

    # 修正索引方式
    batch_indices = torch.arange(valid_pred.size(0), device=valid_pred.device)

    # 手动计算交叉熵loss - 修正索引
    log_softmax = F.log_softmax(valid_pred, dim=1)
    ce_loss = -log_softmax[batch_indices, valid_target.long()]  # 确保target是long类型

    # 计算pt (预测概率)
    pred_prob = F.softmax(valid_pred, dim=1)
    pt = pred_prob[batch_indices, valid_target.long()]  # 同样修正这里

    # Focal Loss: alpha * (1-pt)^gamma * CE
    # 这里使用class_weights作为alpha，为每个类别提供不同的权重
    alpha_per_sample = class_weights[valid_target.long()]  # 确保索引是long类型
    focal_loss_per_sample = alpha_per_sample * (1 - pt) ** gamma * ce_loss

    # 重新构建完整形状，确保数据类型匹配
    voxel_loss_flat = torch.zeros(B * H * W * D, device=pred.device, dtype=focal_loss_per_sample.dtype)
    voxel_loss_flat[valid_mask] = focal_loss_per_sample

    voxel_loss = voxel_loss_flat.reshape(B, H, W, D)
    return voxel_loss



def compute_voxel_wise_ce_loss_fbocc_style(pred, target, class_weights, ignore_index=255):
    """
    按照FBOCC风格实现CE Loss
    """
    B, C, H, W, D = pred.shape

    class_weights = class_weights.to(pred.device)

    # 将pred和target展平
    pred_flat = pred.permute(0, 2, 3, 4, 1).reshape(-1, C)
    target_flat = target.reshape(-1)

    # 过滤ignore区域
    valid_mask = (target_flat != ignore_index)
    valid_pred = pred_flat[valid_mask]
    valid_target = target_flat[valid_mask]

    # 手动计算带权重的交叉熵loss
    log_softmax = F.log_softmax(valid_pred, dim=1)
    ce_loss = -log_softmax[torch.arange(valid_pred.size(0)), valid_target]

    # 应用类别权重
    weight_per_sample = class_weights[valid_target]
    weighted_ce_loss = ce_loss * weight_per_sample

    # 重新构建完整形状
    voxel_loss_flat = torch.zeros(B * H * W * D, device=pred.device)
    voxel_loss_flat[valid_mask] = weighted_ce_loss

    voxel_loss = voxel_loss_flat.reshape(B, H, W, D)
    return voxel_loss


class LossDistributionCalculator(nn.Module):
    def __init__(self, use_focal_loss=True, class_weights=None, gamma=2.0, empty_idx=18,
                 downsample_strategy='nonzero_mean'):
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.class_weights = class_weights
        self.gamma = gamma
        self.empty_idx = empty_idx
        self.downsample_strategy = downsample_strategy

    def forward(self, output_voxels, target_voxels):
        """
        计算每个体素的loss分布和有效掩码
        """
        B, C, H, W, D = output_voxels.shape

        # 计算有效掩码（非255区域）
        valid_mask = (target_voxels != 255)  # [B, H, W, D]

        if self.use_focal_loss:
            loss_dist = compute_voxel_wise_focal_loss_fbocc_style(
                output_voxels, target_voxels,
                class_weights=self.class_weights,
                gamma=self.gamma,
                ignore_index=255
            )
        else:
            loss_dist = compute_voxel_wise_ce_loss_fbocc_style(
                output_voxels, target_voxels,
                class_weights=self.class_weights,
                ignore_index=255
            )

        return loss_dist, valid_mask



class HPNSupervisionLoss(nn.Module):
    """
    HPNet的监督loss：让预测的困难度与真实的loss分布对齐
    """

    def __init__(self, loss_type='kl', temperature=1.0, weight=1.0):
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.weight = weight

    def forward(self, predicted_hardness, true_loss_distribution, valid_mask=None):
        """
        Args:
            predicted_hardness: [B, 100, 100, 8] HPNet预测的困难度
            true_loss_distribution: [B, 100, 100, 8] 真实的loss分布
            valid_mask: [B, 100, 100, 8] 有效区域掩码（True表示有效，False表示255忽略区域）
        """
        B, H, W, D = predicted_hardness.shape

        # 如果没有提供valid_mask，假设所有区域都有效
        if valid_mask is None:
            valid_mask = torch.ones_like(true_loss_distribution, dtype=torch.bool)

        # 确保数值稳定性
        predicted_hardness = torch.clamp(predicted_hardness, min=1e-8, max=1 - 1e-8)
        true_loss_distribution = torch.clamp(true_loss_distribution, min=1e-8)

        # 只考虑有效区域
        pred_valid = predicted_hardness[valid_mask]
        true_valid = true_loss_distribution[valid_mask]

        # 统一归一化到[0,1]范围
        pred_normalized = (pred_valid - pred_valid.min()) / (pred_valid.max() - pred_valid.min() + 1e-8)
        true_normalized = (true_valid - true_valid.min()) / (true_valid.max() - true_valid.min() + 1e-8)

        # 如果没有有效区域，返回0损失
        if pred_valid.numel() == 0:
            return torch.tensor(0.0, device=predicted_hardness.device)

        if self.loss_type == 'kl':
            # KL散度损失 - 只对有效区域计算
            # 归一化有效区域的预测和真实分布
            pred_probs = pred_valid / pred_valid.sum()
            true_probs = F.softmax(true_valid / self.temperature, dim=0)

            loss = F.kl_div(
                torch.log(pred_probs.unsqueeze(0)),
                true_probs.unsqueeze(0),
                reduction='batchmean'
            )

        elif self.loss_type == 'mse':
            # MSE损失 - 只对有效区域计算
            # 先对真实loss进行归一化
            true_norm = true_valid / (true_valid.max() + 1e-8)
            loss = F.mse_loss(pred_valid, true_norm)


        elif self.loss_type == 'focal_correlation':
            weight_map = torch.sigmoid(true_valid * 10)
            loss = (weight_map * (pred_normalized - true_normalized).abs()).mean()

        elif self.loss_type == 'top1_percent_focus':
            # 选择前1%的高loss区域
            k = max(1, len(true_valid) // 100)  # 前1%

            # 找到前1%高loss的索引
            _, topk_indices = torch.topk(true_valid, k)

            # 创建权重图：前1%区域权重为50，其他区域权重为1
            weight_map = torch.ones_like(true_valid)
            weight_map[topk_indices] = 50.0  # 给前1%区域50倍权重

            # 计算加权损失
            loss = (weight_map * (pred_normalized - true_normalized).abs()).mean()

        elif self.loss_type == 'multi_level_focus_onlyloss':
            # 您的改进版本：多级关注策略
            n_total = len(true_valid)
            k1 = max(1, n_total // 100)  # 前1%
            k2 = max(1, n_total // 50)  # 前2%

            # ===== 第一级：前1%区域 =====
            # 1. 前1%高loss的索引
            _, top1_loss_indices = torch.topk(true_valid, k1)


            # ===== 第二级：前1%-2%区域 ====
            # 1. 前2%高loss的索引（排除前1%）
            _, top2_loss_indices_all = torch.topk(true_valid, k2)
            top2_loss_indices = top2_loss_indices_all[k1:]  # 取第1%-2%


            # ===== 创建多级权重图 =====
            weight_map = torch.ones_like(true_valid)

            # 第一级：前1%并集区域，权重50
            weight_map[top1_loss_indices] = 50.0

            # 第二级：前1%-2%并集区域，权重25
            # 注意：确保第二级不与第一级重叠（虽然理论上不应该重叠）
            weight_map[top2_loss_indices] = 10.0

            # 计算加权损失
            loss = (weight_map * (pred_normalized - true_normalized).abs()).mean()



        elif self.loss_type == 'multi_level_focus':
            # 您的改进版本：多级关注策略
            n_total = len(true_valid)
            k1 = max(1, n_total // 100)  # 前1%
            k2 = max(1, n_total // 50)  # 前2%

            # ===== 第一级：前1%区域 =====
            # 1. 前1%高loss的索引
            _, top1_loss_indices = torch.topk(true_valid, k1)

            # 2. 前1%高预测困难度的索引
            _, top1_pred_indices = torch.topk(pred_valid, k1)

            # 3. 取并集
            top1_union_indices = torch.unique(torch.cat([top1_loss_indices, top1_pred_indices]))

            # ===== 第二级：前1%-2%区域 =====
            # 1. 前2%高loss的索引（排除前1%）
            _, top2_loss_indices_all = torch.topk(true_valid, k2)
            top2_loss_indices = top2_loss_indices_all[k1:]  # 取第1%-2%

            # 2. 前2%高预测困难度的索引（排除前1%）
            _, top2_pred_indices_all = torch.topk(pred_valid, k2)
            top2_pred_indices = top2_pred_indices_all[k1:]  # 取第1%-2%

            # 3. 取并集
            top2_union_indices = torch.unique(torch.cat([top2_loss_indices, top2_pred_indices]))

            # ===== 创建多级权重图 =====
            weight_map = torch.ones_like(true_valid)

            # 第一级：前1%并集区域，权重50
            weight_map[top1_union_indices] = 50.0

            # 第二级：前1%-2%并集区域，权重25
            # 注意：确保第二级不与第一级重叠（虽然理论上不应该重叠）
            weight_map[top2_union_indices] = 5.0

            # 计算加权损失
            loss = (weight_map * (pred_normalized - true_normalized).abs()).mean()

        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        return loss * self.weight