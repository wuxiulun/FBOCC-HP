#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量统计分析监控统计数据
功能：遍历所有样本的_stats.json文件，计算总体统计量并生成报告
"""

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class MonitorStatsAnalyzer:
    def __init__(self, stats_dir):
        """
        初始化分析器

        Args:
            stats_dir: monitor_stats目录路径
        """
        self.stats_dir = stats_dir
        self.all_stats = []
        self.df = None

    def load_all_stats(self):
        """加载所有样本的统计数据"""
        print(f"正在从 {self.stats_dir} 加载统计数据...")

        self.all_stats = []
        sample_count = 0

        # 遍历目录中的所有JSON文件
        for filename in os.listdir(self.stats_dir):
            if filename.endswith('_stats.json'):
                filepath = os.path.join(self.stats_dir, filename)

                try:
                    with open(filepath, 'r') as f:
                        stats_data = json.load(f)

                    # 添加文件名信息
                    stats_data['filename'] = filename
                    sample_token = filename.replace('_stats.json', '')
                    stats_data['sample_token'] = sample_token

                    self.all_stats.append(stats_data)
                    sample_count += 1

                except Exception as e:
                    print(f"加载文件 {filename} 时出错: {e}")

        print(f"成功加载 {sample_count} 个样本的统计数据")
        return sample_count

    def create_dataframe(self):
        """将统计数据转换为DataFrame"""
        if not self.all_stats:
            print("没有可用的统计数据")
            return None

        # 创建DataFrame
        self.df = pd.DataFrame(self.all_stats)

        # 确保数值列的类型正确
        numeric_columns = [
            'valid_voxel_count', 'overall_loss_mean',
            'top1_hardness_loss_mean', 'top1_scene_loss_mean', 'top1_global_loss_mean',
            'hardness_relative_gain', 'scene_relative_gain', 'global_relative_gain',
            'hardness_pearson_corr', 'scene_pearson_corr', 'global_pearson_corr',
            'hardness_spearman_corr', 'scene_spearman_corr', 'global_spearman_corr',
            'hardness_precision', 'scene_precision', 'global_precision',
            'hardness_iou', 'scene_iou', 'global_iou'
        ]

        # 只保留实际存在的列
        existing_numeric_cols = [col for col in numeric_columns if col in self.df.columns]
        for col in existing_numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        print(f"DataFrame 创建完成，包含 {len(self.df)} 行，{len(self.df.columns)} 列")
        return self.df

    def compute_summary_statistics(self):
        """计算总体统计量"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return None

        summary = {}

        # 需要分析的数值列
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        print("\n" + "=" * 80)
        print("总体统计分析")
        print("=" * 80)

        for col in numeric_cols:
            if col in ['valid_voxel_count', 'filename']:
                continue

            values = self.df[col].dropna()
            if len(values) > 0:
                summary[f"{col}_count"] = len(values)
                summary[f"{col}_mean"] = values.mean()
                summary[f"{col}_std"] = values.std()
                summary[f"{col}_min"] = values.min()
                summary[f"{col}_max"] = values.max()
                summary[f"{col}_median"] = values.median()
                summary[f"{col}_q25"] = values.quantile(0.25)
                summary[f"{col}_q75"] = values.quantile(0.75)

                # 打印关键统计指标
                if any(keyword in col for keyword in ['loss_mean', '_corr', '_gain', '_precision', '_iou']):
                    print(f"\n{col}:")
                    print(f"  均值: {values.mean():.4f} ± {values.std():.4f}")
                    print(f"  范围: [{values.min():.4f}, {values.max():.4f}]")
                    print(f"  中位数: {values.median():.4f}")

        # 样本数统计
        summary['total_samples'] = len(self.df)

        # 计算各指标的缺失率
        missing_rates = {}
        for col in numeric_cols:
            missing_rate = self.df[col].isna().mean() * 100
            if missing_rate > 0:
                missing_rates[col] = missing_rate

        if missing_rates:
            print(f"\n缺失值统计:")
            for col, rate in sorted(missing_rates.items(), key=lambda x: x[1], reverse=True):
                print(f"  {col}: {rate:.1f}%")

        return summary

    def analyze_correlations(self):
        """分析不同指标之间的相关性"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return None

        # 选择要分析相关性的指标
        correlation_metrics = [
            'overall_loss_mean',
            'hardness_pearson_corr', 'scene_pearson_corr', 'global_pearson_corr',
            'hardness_precision', 'scene_precision', 'global_precision',
            'hardness_iou', 'scene_iou', 'global_iou'
        ]

        # 只保留实际存在的列
        existing_metrics = [col for col in correlation_metrics if col in self.df.columns]

        if len(existing_metrics) >= 2:
            print("\n" + "=" * 80)
            print("指标间相关性分析")
            print("=" * 80)

            # 计算相关性矩阵
            corr_matrix = self.df[existing_metrics].corr()

            # 打印关键相关性
            print("\n关键相关性:")

            # 损失与困难度识别的相关性
            if 'overall_loss_mean' in existing_metrics:
                for metric in ['hardness_pearson_corr', 'scene_pearson_corr', 'global_pearson_corr']:
                    if metric in existing_metrics:
                        corr = corr_matrix.loc['overall_loss_mean', metric]
                        print(f"  总体损失 vs {metric}: {corr:.3f}")

            # 困难度识别指标之间的相关性
            hard_metrics = [m for m in ['hardness', 'scene', 'global'] if f'{m}_pearson_corr' in existing_metrics]
            for i in range(len(hard_metrics)):
                for j in range(i + 1, len(hard_metrics)):
                    m1 = f'{hard_metrics[i]}_pearson_corr'
                    m2 = f'{hard_metrics[j]}_pearson_corr'
                    corr = corr_matrix.loc[m1, m2]
                    print(f"  {m1} vs {m2}: {corr:.3f}")

            return corr_matrix
        return None

    def analyze_hardness_effectiveness(self):
        """分析困难度识别的有效性"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return None

        print("\n" + "=" * 80)
        print("困难度识别有效性分析")
        print("=" * 80)

        # 分析不同困难度方法的提升效果
        gain_metrics = []
        if 'hardness_relative_gain' in self.df.columns:
            gain_metrics.append(('体素级困难度', 'hardness_relative_gain'))
        if 'scene_relative_gain' in self.df.columns:
            gain_metrics.append(('场景级困难度', 'scene_relative_gain'))
        if 'global_relative_gain' in self.df.columns:
            gain_metrics.append(('全局困难度', 'global_relative_gain'))

        if gain_metrics:
            print("\n相对提升百分比:")
            for name, col in gain_metrics:
                values = self.df[col].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    # 转换百分比显示
                    print(f"  {name}: {mean_val:.1f}% ± {std_val:.1f}%")
                    print(f"    范围: [{values.min():.1f}%, {values.max():.1f}%]")

        # 分析精确率和IoU
        precision_metrics = []
        for prefix in ['hardness', 'scene', 'global']:
            precision_col = f'{prefix}_precision'
            iou_col = f'{prefix}_iou'
            if precision_col in self.df.columns and iou_col in self.df.columns:
                precision_vals = self.df[precision_col].dropna()
                iou_vals = self.df[iou_col].dropna()

                if len(precision_vals) > 0 and len(iou_vals) > 0:
                    precision_metrics.append((
                        f'{prefix}_precision',
                        f'{prefix}_iou',
                        precision_vals.mean(),
                        iou_vals.mean()
                    ))

        if precision_metrics:
            print("\n困难度识别精确率 (Precision) 和 IoU:")
            for prec_col, iou_col, prec_mean, iou_mean in precision_metrics:
                print(f"  {prec_col}: {prec_mean:.3f}")
                print(f"  {iou_col}: {iou_mean:.3f}")

    def identify_extreme_samples(self, top_n=10):
        """识别极端样本（表现最好/最差的样本）"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return None

        print("\n" + "=" * 80)
        print(f"极端样本分析 (前{top_n}个)")
        print("=" * 80)

        # 定义要分析的指标
        analysis_metrics = [
            ('overall_loss_mean', '总体损失', 'ascending'),  # 损失越小越好
            ('hardness_pearson_corr', '体素级相关性', 'descending'),  # 相关性越高越好
            ('hardness_precision', '体素级精确率', 'descending'),  # 精确率越高越好
            ('hardness_relative_gain', '体素级相对提升', 'descending'),  # 提升越大越好
        ]

        # 只保留实际存在的指标
        analysis_metrics = [(col, name, order) for col, name, order in analysis_metrics
                            if col in self.df.columns]

        for col, name, order in analysis_metrics:
            if order == 'ascending':
                # 找出值最小的样本（损失小）
                top_samples = self.df.nsmallest(top_n, col)[['sample_token', col]]
                print(f"\n{name} 最好的样本 (值最小):")
            else:
                # 找出值最大的样本（相关性高、精确率高、提升大）
                top_samples = self.df.nlargest(top_n, col)[['sample_token', col]]
                print(f"\n{name} 最好的样本 (值最大):")

            for idx, row in top_samples.iterrows():
                print(f"  样本 {row['sample_token']}: {row[col]:.4f}")

    def create_visualizations(self, output_dir):
        """创建可视化图表"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. 损失分布直方图
        if 'overall_loss_mean' in self.df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(self.df['overall_loss_mean'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('总体平均损失')
            plt.ylabel('样本数量')
            plt.title('总体损失分布')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'loss_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # 2. 相关性指标箱线图
        corr_cols = [col for col in self.df.columns if 'corr' in col]
        if corr_cols:
            plt.figure(figsize=(12, 6))
            data_to_plot = [self.df[col].dropna() for col in corr_cols]
            plt.boxplot(data_to_plot, labels=corr_cols)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('相关系数')
            plt.title('不同困难度与损失的相关性分布')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_boxplot.png'), dpi=150, bbox_inches='tight')
            plt.close()

        # 3. 相对提升百分比条形图
        gain_cols = [col for col in self.df.columns if 'gain' in col]
        if gain_cols:
            plt.figure(figsize=(10, 6))
            means = [self.df[col].mean() for col in gain_cols]
            stds = [self.df[col].std() for col in gain_cols]

            x_pos = range(len(gain_cols))
            plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
            plt.xticks(x_pos, [col.replace('_relative_gain', '') for col in gain_cols], rotation=45)
            plt.ylabel('相对提升百分比 (%)')
            plt.title('不同困难度方法的相对提升效果')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'relative_gain_barplot.png'), dpi=150, bbox_inches='tight')
            plt.close()

        print(f"可视化图表已保存到: {output_dir}")

    def save_summary_report(self, output_dir):
        """保存汇总报告"""
        if self.df is None:
            print("请先加载数据并创建DataFrame")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存详细数据到CSV
        csv_path = os.path.join(output_dir, 'all_samples_stats.csv')
        self.df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"详细统计数据已保存到: {csv_path}")

        # 2. 计算并保存汇总统计
        summary = self.compute_summary_statistics()
        if summary:
            summary_df = pd.DataFrame([summary])
            summary_csv_path = os.path.join(output_dir, 'summary_statistics.csv')
            summary_df.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            print(f"汇总统计数据已保存到: {summary_csv_path}")

        # 3. 保存文本报告
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("监控统计数据批量分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"样本总数: {len(self.df)}\n")
            f.write(f"统计指标数: {len(self.df.columns)}\n\n")

            # 写入基本统计
            f.write("=" * 80 + "\n")
            f.write("基本统计分析\n")
            f.write("=" * 80 + "\n\n")

            basic_metrics = ['valid_voxel_count', 'overall_loss_mean']
            for metric in basic_metrics:
                if metric in self.df.columns:
                    values = self.df[metric].dropna()
                    if len(values) > 0:
                        f.write(f"{metric}:\n")
                        f.write(f"  均值: {values.mean():.4f}\n")
                        f.write(f"  标准差: {values.std():.4f}\n")
                        f.write(f"  范围: [{values.min():.4f}, {values.max():.4f}]\n")
                        f.write(f"  中位数: {values.median():.4f}\n\n")

            # 写入相关性分析
            if any('corr' in col for col in self.df.columns):
                f.write("=" * 80 + "\n")
                f.write("困难度与损失相关性分析\n")
                f.write("=" * 80 + "\n\n")

                for prefix in ['hardness', 'scene', 'global']:
                    pearson_col = f'{prefix}_pearson_corr'
                    spearman_col = f'{prefix}_spearman_corr'

                    if pearson_col in self.df.columns:
                        pearson_vals = self.df[pearson_col].dropna()
                        if len(pearson_vals) > 0:
                            f.write(f"{prefix}皮尔逊相关性:\n")
                            f.write(f"  均值: {pearson_vals.mean():.4f}\n")
                            f.write(f"  标准差: {pearson_vals.std():.4f}\n\n")

                    if spearman_col in self.df.columns:
                        spearman_vals = self.df[spearman_col].dropna()
                        if len(spearman_vals) > 0:
                            f.write(f"{prefix}斯皮尔曼相关性:\n")
                            f.write(f"  均值: {spearman_vals.mean():.4f}\n")
                            f.write(f"  标准差: {spearman_vals.std():.4f}\n\n")

            # 写入困难度识别效果
            if any('gain' in col for col in self.df.columns):
                f.write("=" * 80 + "\n")
                f.write("困难度识别效果分析\n")
                f.write("=" * 80 + "\n\n")

                for prefix in ['hardness', 'scene', 'global']:
                    gain_col = f'{prefix}_relative_gain'
                    if gain_col in self.df.columns:
                        gain_vals = self.df[gain_col].dropna()
                        if len(gain_vals) > 0:
                            f.write(f"{prefix}相对提升:\n")
                            f.write(f"  均值: {gain_vals.mean():.1f}%\n")
                            f.write(f"  标准差: {gain_vals.std():.1f}%\n\n")

            # 写入样本列表
            f.write("=" * 80 + "\n")
            f.write("样本列表\n")
            f.write("=" * 80 + "\n\n")

            for idx, row in self.df.iterrows():
                sample_info = f"样本 {row.get('sample_token', 'unknown')}: "
                if 'overall_loss_mean' in row and not pd.isna(row['overall_loss_mean']):
                    sample_info += f"损失={row['overall_loss_mean']:.4f}"
                if 'hardness_pearson_corr' in row and not pd.isna(row['hardness_pearson_corr']):
                    sample_info += f", 体素相关性={row['hardness_pearson_corr']:.3f}"
                f.write(sample_info + "\n")

        print(f"分析报告已保存到: {report_path}")

    def run_complete_analysis(self, output_dir):
        """运行完整的分析流程"""
        print("开始批量统计分析...")

        # 1. 加载数据
        sample_count = self.load_all_stats()
        if sample_count == 0:
            print("没有找到统计数据，分析终止")
            return

        # 2. 创建DataFrame
        self.create_dataframe()

        # 3. 计算总体统计量
        print("\n" + "=" * 80)
        print("开始计算总体统计量")
        print("=" * 80)

        summary = self.compute_summary_statistics()

        # 4. 分析相关性
        self.analyze_correlations()

        # 5. 分析困难度识别有效性
        self.analyze_hardness_effectiveness()

        # 6. 识别极端样本
        self.identify_extreme_samples(top_n=5)

        # 7. 创建可视化图表
        viz_dir = os.path.join(output_dir, 'visualizations')
        self.create_visualizations(viz_dir)

        # 8. 保存汇总报告
        self.save_summary_report(output_dir)

        print("\n" + "=" * 80)
        print("分析完成！")
        print(f"结果已保存到: {output_dir}")
        print("=" * 80)


# 使用示例
def main():
    # 配置路径
    # 假设您的目录结构如下：
    # test/
    #   ├── monitor_stats/          # 包含所有_stats.json文件
    #   ├── monitor_signals/
    #   ├── occupancy_pred/
    #   └── results.csv

    test_dir = "/root/autodl-tmp/test/fbocc-r50-cbgs_depth_16f_16x4_20e/Fri_Jan__9_19_05"  # 请修改为您的test文件夹路径
    stats_dir = os.path.join(test_dir, "monitor_stats")
    output_dir = os.path.join(test_dir, "analysis_results")

    # 创建分析器并运行分析
    analyzer = MonitorStatsAnalyzer(stats_dir)
    analyzer.run_complete_analysis(output_dir)


if __name__ == "__main__":
    main()