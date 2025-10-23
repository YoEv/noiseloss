import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from scipy.stats import pearsonr, spearmanr

# 配置
DATASETS = ['asap', 'shutter', 'unconditional']
MODELS = ['small', 'medium', 'mgen-melody', 'large']
TOKEN_LENGTHS = [5, 10, 50, 100, 150, 200]

# 颜色（与white_noise脚本保持一致）
BASE_MACARON_COLORS = {
    'reference': '#4CAF50',
    'white': '#E8E8E8',
    'asap': '#E64A19',        # 深橙
    'shutter': '#7B1FA2',     # 紫色  
    'unconditional': '#1565C0' # 蓝色
    # 'asap': '#FFE082',
    # 'shutter': '#F8BBD9',
    # 'unconditional': '#B3E5FC'
}

# 顶部的常量和颜色保持不变
def generate_gradient_colors(base_color, num_steps=6):
    base_rgb = to_rgb(base_color)
    colors = []
    for i in range(num_steps):
        factor = 0.9 - (i * 0.5 / (num_steps - 1))
        darker_rgb = tuple(c * factor for c in base_rgb)
        colors.append(darker_rgb)
    return colors

DATASET_GRADIENT_COLORS = {ds: generate_gradient_colors(BASE_MACARON_COLORS[ds]) for ds in DATASETS}

# 添加线型和标记符号映射
DATASET_LINESTYLES = {
    'asap': '-',      # 实线
    'shutter': '--',  # 虚线
    'unconditional': '-.'  # 点划线
}

DATASET_MARKERS = {
    'asap': 'o',      # 圆圈
    'shutter': 's',   # 方块
    'unconditional': '^'  # 三角形
}

def get_base_dir(model):
    if model == 'large':
        return '/home/evev/asap-dataset/+Loss/Phase5_1_Large/Phase5_1'
    else:
        return '/home/evev/asap-dataset/+Loss/Phase5_1'

# 新增：分别为参考集与噪声集提供前缀候选，匹配真实目录命名
def ref_prefix_candidates(dataset):
    if dataset == 'asap':
        return ['asap']
    elif dataset == 'shutter':
        return ['ShutterStock_32k']
    elif dataset == 'unconditional':
        return ['Unconditional']
    else:
        return [dataset]

def noise_prefix_candidates(dataset):
    if dataset == 'asap':
        return ['asap']
    elif dataset == 'shutter':
        return ['shutter']
    elif dataset == 'unconditional':
        return ['unconditional']
    else:
        return [dataset]

def find_existing_dir(base_dir, candidates):
    """
    在 base_dir 下按候选目录名顺序查找第一个存在的目录，找到即返回完整路径，否则返回 None。
    """
    for d in candidates:
        full = os.path.join(base_dir, d)
        if os.path.isdir(full):
            return full
    return None

def find_existing_file(base_dir, candidates, filename='results.txt'):
    """
    在 base_dir 下按候选目录名顺序查找包含 filename 的第一个存在的文件，返回文件完整路径，否则 None。
    """
    for d in candidates:
        full_dir = os.path.join(base_dir, d)
        full_file = os.path.join(full_dir, filename)
        if os.path.isfile(full_file):
            return full_file
    return None

def ref_dir_candidates(dataset, model):
    # {prefix}_ori_{model}，prefix 依数据集不同
    prefixes = ref_prefix_candidates(dataset)
    return [f'{p}_ori_{model}' for p in prefixes]

def noise_dir_candidates(dataset, model, tk):
    # {prefix}_replace_noise_white_at5_tk{tk}_token_loss_{model}，prefix 依数据集不同
    prefixes = noise_prefix_candidates(dataset)
    return [f'{p}_replace_noise_white_at5_tk{tk}_token_loss_{model}' for p in prefixes]

def read_results_txt(file_path):
    """
    读取 results.txt，返回 List[Tuple[str, float]]
    兼容多种格式：
      1) key: value
      2) key\\tvalue   （含中文表头“文件名\\t损失值”会被跳过）
      3) 若上面都不匹配，按空白分隔，取最后一个字段为数值，其余作为 key
    注意：此处不做 basename 规范化，也不去重；交给后续聚合函数处理。
    """
    if not file_path or not os.path.exists(file_path):
        return None
    entries = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            # 跳过表头
            if line.startswith('文件名') or line.lower().startswith('filename'):
                continue

            # 1) 冒号分隔
            if ':' in line:
                key, val = line.split(':', 1)
                key = key.strip()
                try:
                    v = float(val.strip())
                    entries.append((key, v))
                    continue
                except Exception:
                    pass

            # 2) 制表符分隔
            parts = [p for p in line.split('\t') if p != '']
            if len(parts) >= 2:
                key = parts[0].strip()
                val_str = parts[-1].strip()
                try:
                    v = float(val_str)
                    entries.append((key, v))
                    continue
                except Exception:
                    pass

            # 3) 空白分隔，取最后一个字段为数值
            parts = line.split()
            if len(parts) >= 2:
                key = ' '.join(parts[:-1]).strip()
                val_str = parts[-1].strip()
                try:
                    v = float(val_str)
                    entries.append((key, v))
                except Exception:
                    pass

    return entries

def _normalize_key(dataset, key):
    """
    将不同来源、不同路径格式的 key 统一为同一个"规范键"。
    现在所有数据都应该有正确的格式，直接使用原始key。
    """
    return key.strip()

def _aggregate_by_canonical_key(dataset, raw):
    """
    将原始读入的 (key, value) 列表按规范键聚合
    返回：{canonical_key: mean_value}
    """
    if not raw:
        return {}

    bucket = {}
    # 兼容 list[(k, v)] 和 dict{k: v}
    if isinstance(raw, dict):
        items = raw.items()
    else:
        items = raw

    for k, v in items:
        ck = _normalize_key(dataset, k)
        bucket.setdefault(ck, []).append(float(v))

    # 取均值（处理重复键）
    return {ck: float(np.mean(vals)) for ck, vals in bucket.items()}

def compute_diff_stats_for(model, dataset):
    """
    使用results.txt计算：每个token长度下，(L' - L) 的均值与标准差（跨歌曲）
    同时返回reference均值（用于标注）
    返回结构：
    {
      'reference_mean': float,
      'noise_diff': {
        'white': {
          'tk5': {'mean': float, 'std': float, 'count': int},
          ...
        }
      }
    }
    """
    base_dir = get_base_dir(model)

    # reference results
    ref_candidates = ref_dir_candidates(dataset, model)
    ref_file = find_existing_file(base_dir, ref_candidates, filename='results.txt')
    if ref_file is None:
        print(f'[WARN] Missing reference results for {dataset}-{model}. Tried: {[os.path.join(base_dir, d, "results.txt") for d in ref_candidates]}')
        ref_map_raw = {}
    else:
        ref_map_raw = read_results_txt(ref_file)

    # 规范化 + 重复合并
    ref_map = _aggregate_by_canonical_key(dataset, ref_map_raw)
    ref_values = list(ref_map.values())
    reference_mean = float(np.mean(ref_values)) if ref_values else float('nan')
    reference_std = float(np.std(ref_values)) if ref_values else float('nan')  # 添加标准差计算

    out = {
        'reference_mean': reference_mean,
        'reference_std': reference_std,  # 添加到输出结构
        'noise_diff': {'white': {}}
    }

    for tk in TOKEN_LENGTHS:
        noise_candidates = noise_dir_candidates(dataset, model, tk)
        noise_file = find_existing_file(base_dir, noise_candidates, filename='results.txt')
        if noise_file is None:
            print(f'[WARN] Missing noise results for {dataset}-{model}-tk{tk}. Tried: {[os.path.join(base_dir, d, "results.txt") for d in noise_candidates]}')
            out['noise_diff']['white'][f'tk{tk}'] = {'mean': float('nan'), 'std': float('nan'), 'count': 0}
            continue

        noise_map_raw = read_results_txt(noise_file) or {}
        noise_map = _aggregate_by_canonical_key(dataset, noise_map_raw)

        # 对齐歌曲集合（使用规范化后的 key）
        common_keys = set(ref_map.keys()).intersection(noise_map.keys())
        diffs = []
        for k in common_keys:
            L = ref_map[k]
            Lp = noise_map[k]
            diffs.append(Lp - L)

        if len(diffs) == 0:
            out['noise_diff']['white'][f'tk{tk}'] = {'mean': float('nan'), 'std': float('nan'), 'count': 0}
        else:
            diffs_arr = np.array(diffs, dtype=float)
            mean_diff = float(np.mean(diffs_arr))
            std_diff = float(np.std(diffs_arr))  # 跨歌曲的标准差
            out['noise_diff']['white'][f'tk{tk}'] = {'mean': mean_diff, 'std': std_diff, 'count': len(diffs)}

    return out

def collect_all_models_data():
    """
    构建绘图所需数据：
    all_model_data[model][dataset] = {
      'reference_mean': float,
      'noise_diff': {'white': {'tkX': {'mean':, 'std':, 'count':}}}
    }
    """
    all_data = {}
    for model in MODELS:
        all_data[model] = {}
        for dataset in DATASETS:
            all_data[model][dataset] = compute_diff_stats_for(model, dataset)
    return all_data

def perform_token_length_correlation_analysis(base_dir, models, databases, token_lengths):
    """
    分析噪声loss与token length的相关性
    检验假设：随着token length增加，噪声loss显著降低（负相关）
    """
    import numpy as np
    from scipy import stats
    
    results = []
    
    for model in models:
        print(f"\n处理模型: {model}")
        model_base_dir = get_base_dir(model)
        
        if not os.path.exists(model_base_dir):
            print(f"警告: 模型路径不存在: {model_base_dir}")
            continue
            
        for database in databases:
            print(f"  处理数据库: {database}")
            
            # 收集该模型-数据库组合下所有token length的噪声loss数据
            token_length_data = []
            loss_data = []
            
            for token_length in token_lengths:
                # 构建噪声数据路径（固定白噪声）
                noise_candidates = noise_dir_candidates_with_color(database, model, token_length, 'white')
                noise_file = find_existing_file(model_base_dir, noise_candidates, 'results.txt')
                
                if not noise_file:
                    print(f"    警告: Token长度{token_length}的噪声数据文件不存在")
                    continue
                
                try:
                    # 读取并聚合噪声数据
                    noise_data = read_results_txt(noise_file)
                    if not noise_data:
                        continue
                    
                    noise_aggregated = _aggregate_by_canonical_key(database, noise_data)
                    
                    # 计算该token length下的平均loss
                    if noise_aggregated:
                        avg_loss = np.mean(list(noise_aggregated.values()))
                        token_length_data.append(token_length)
                        loss_data.append(avg_loss)
                        print(f"    Token长度{token_length}: 平均loss={avg_loss:.4f}, 歌曲数={len(noise_aggregated)}")
                    
                except Exception as e:
                    print(f"    错误: Token长度{token_length}数据处理失败: {e}")
                    continue
            
            # 检查数据点是否足够进行相关性分析
            if len(token_length_data) < 3:
                print(f"    警告: {database}数据点不足({len(token_length_data)}个), 跳过相关性分析")
                continue
            
            # 计算相关性分析
            try:
                # Pearson相关性（线性关系）
                pearson_corr, pearson_p = stats.pearsonr(token_length_data, loss_data)
                
                # Spearman相关性（单调关系）
                spearman_corr, spearman_p = stats.spearmanr(token_length_data, loss_data)
                
                # 单侧检验：检验是否显著负相关（H1: r < 0）
                # 对于负相关假设，单侧p值 = 双侧p值/2 (当相关系数<0时)
                pearson_p_onetailed = pearson_p / 2 if pearson_corr < 0 else 1 - (pearson_p / 2)
                spearman_p_onetailed = spearman_p / 2 if spearman_corr < 0 else 1 - (spearman_p / 2)
                
                # 线性回归分析（获取斜率和截距）
                slope, intercept, r_value, p_value, std_err = stats.linregress(token_length_data, loss_data)
                
                result = {
                    'model': model,
                    'database': database,
                    'n_token_lengths': len(token_length_data),
                    'token_lengths': token_length_data,
                    'avg_losses': loss_data,
                    
                    # Pearson相关性
                    'pearson_correlation': pearson_corr,
                    'pearson_p_twotailed': pearson_p,
                    'pearson_p_onetailed': pearson_p_onetailed,
                    
                    # Spearman相关性
                    'spearman_correlation': spearman_corr,
                    'spearman_p_twotailed': spearman_p,
                    'spearman_p_onetailed': spearman_p_onetailed,
                    
                    # 线性回归
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'slope_p_value': p_value,
                    'slope_std_error': std_err,
                    
                    # 显著性判断（α=0.05）
                    'is_significant_negative_pearson': (pearson_corr < 0) and (pearson_p_onetailed < 0.05),
                    'is_significant_negative_spearman': (spearman_corr < 0) and (spearman_p_onetailed < 0.05),
                    'is_significant_negative_slope': (slope < 0) and (p_value < 0.05)
                }
                
                results.append(result)
                
                # 输出结果
                print(f"    相关性分析结果:")
                print(f"      Pearson: r={pearson_corr:.4f}, p(单侧)={pearson_p_onetailed:.4f}")
                print(f"      Spearman: ρ={spearman_corr:.4f}, p(单侧)={spearman_p_onetailed:.4f}")
                print(f"      线性回归: 斜率={slope:.6f}, p={p_value:.4f}")
                
                if result['is_significant_negative_pearson'] or result['is_significant_negative_spearman']:
                    print(f"      ✓ 检测到显著负相关！")
                else:
                    print(f"      ✗ 未检测到显著负相关")
                    
            except Exception as e:
                print(f"    错误: 相关性计算失败: {e}")
                continue
    
    # 保存详细结果
    if results:
        # 保存到CSV
        results_df = pd.DataFrame(results)
        output_file = os.path.join('/home/evev/asap-dataset/+Loss_Plot/Phase5_1/Figure2', 'token_length_correlation_analysis.csv')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        results_df.to_csv(output_file, index=False)
        
        # 生成可读的分析报告
        report_file = os.path.join('/home/evev/asap-dataset/+Loss_Plot/Phase5_1/Figure2', 'token_length_correlation_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Token Length vs Noise Loss 负相关分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write("研究假设: 随着token length增加，噪声loss显著降低\n\n")
            
            # 总体摘要
            total_analyses = len(results)
            significant_pearson = sum(1 for r in results if r['is_significant_negative_pearson'])
            significant_spearman = sum(1 for r in results if r['is_significant_negative_spearman'])
            significant_slope = sum(1 for r in results if r['is_significant_negative_slope'])
            
            f.write(f"总体结果摘要:\n")
            f.write(f"总分析数: {total_analyses}\n")
            f.write(f"显著负相关(Pearson): {significant_pearson}/{total_analyses} ({significant_pearson/total_analyses*100:.1f}%)\n")
            f.write(f"显著负相关(Spearman): {significant_spearman}/{total_analyses} ({significant_spearman/total_analyses*100:.1f}%)\n")
            f.write(f"显著负斜率: {significant_slope}/{total_analyses} ({significant_slope/total_analyses*100:.1f}%)\n\n")
            
            # 详细结果
            for result in results:
                f.write(f"模型: {result['model'].upper()}\n")
                f.write(f"数据库: {result['database'].upper()}\n")
                f.write(f"数据点数: {result['n_token_lengths']}\n")
                f.write(f"Token长度范围: {min(result['token_lengths'])}-{max(result['token_lengths'])}\n")
                f.write(f"\n相关性分析:\n")
                f.write(f"  Pearson相关系数: {result['pearson_correlation']:.4f}\n")
                f.write(f"  Pearson单侧p值: {result['pearson_p_onetailed']:.6f}\n")
                f.write(f"  Spearman相关系数: {result['spearman_correlation']:.4f}\n")
                f.write(f"  Spearman单侧p值: {result['spearman_p_onetailed']:.6f}\n")
                f.write(f"\n线性回归分析:\n")
                f.write(f"  斜率: {result['slope']:.6f}\n")
                f.write(f"  截距: {result['intercept']:.4f}\n")
                f.write(f"  R²: {result['r_squared']:.4f}\n")
                f.write(f"  斜率p值: {result['slope_p_value']:.6f}\n")
                f.write(f"\n显著性检验结果 (α=0.05):\n")
                f.write(f"  Pearson负相关显著: {'是' if result['is_significant_negative_pearson'] else '否'}\n")
                f.write(f"  Spearman负相关显著: {'是' if result['is_significant_negative_spearman'] else '否'}\n")
                f.write(f"  线性负斜率显著: {'是' if result['is_significant_negative_slope'] else '否'}\n")
                f.write("-" * 40 + "\n\n")
        
        print(f"\n分析完成！")
        print(f"详细结果已保存到: {output_file}")
        print(f"分析报告已保存到: {report_file}")
        print(f"\n快速摘要:")
        print(f"  总分析数: {total_analyses}")
        print(f"  显著负相关(Pearson): {significant_pearson}/{total_analyses}")
        print(f"  显著负相关(Spearman): {significant_spearman}/{total_analyses}")
        
    else:
        print("\n警告: 没有成功计算任何相关性结果")
    
    return results

def noise_dir_candidates_with_color(dataset, model, tk, noise_color):
    """
    扩展原有的noise_dir_candidates函数，支持不同噪声颜色
    """
    prefixes = noise_prefix_candidates(dataset)
    return [f'{p}_replace_noise_{noise_color}_at5_tk{tk}_token_loss_{model}' for p in prefixes]

def create_line_plot_combined_single(all_model_data, output_dir, suffix='white_std'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'combined_line_plot_{suffix}_name_2.png')

    fig, axes = plt.subplots(1, 4, figsize=(32, 6.8))
    # 设置透明背景
    fig.patch.set_alpha(0.0)
    
    # fig.suptitle('Combined Line Plot: All Models - White Noise (Std across song diffs)', fontsize=20, fontweight='bold')

    for model_idx, model in enumerate(MODELS):
        ax = axes[model_idx]
        # 设置子图背景透明
        ax.patch.set_alpha(0.0)
        # ax.set_title(f'{model.title()} Model', fontsize=16, fontweight='bold')

        # reference文本收集
        reference_texts = []

        for dataset_idx, dataset in enumerate(DATASETS):
            dataset_data = all_model_data[model][dataset]

            means = []
            stds = []
            for tk in TOKEN_LENGTHS:
                stat = dataset_data['noise_diff']['white'].get(f'tk{tk}', {'mean': float('nan'), 'std': float('nan')})
                means.append(stat['mean'])
                stds.append(stat['std'])

            x = list(range(len(TOKEN_LENGTHS)))
            color = DATASET_GRADIENT_COLORS[dataset][0]
            linestyle = DATASET_LINESTYLES[dataset]  # 获取线型
            marker = DATASET_MARKERS[dataset]        # 获取标记符号

            # 画误差棒与散点
            # 为每个数据集添加小的x轴偏移，避免点重叠
            offset = (dataset_idx - 1) * 0.2  # dataset_idx是数据集的索引，范围通常是0,1,2
            x_offset = [xi + offset for xi in x]  # 使用列表推导式计算偏移
            
            ax.errorbar(
                x_offset, means, yerr=stds,
                marker=marker, linewidth=3, markersize=10,  # 使用对应的标记符号
                linestyle=linestyle,  # 使用对应的线型
                capsize=8, capthick=2, label=dataset.title(), color=color, alpha=0.72  # 添加透明度
            )
            ax.scatter(x_offset, means, s=20, color=color, zorder=30, marker=marker, alpha=0.72)  # scatter也添加透明度

            # 参考值展示：reference均值±标准差
            ref_mean = dataset_data.get('reference_mean', float('nan'))
            ref_std = dataset_data.get('reference_std', float('nan'))  # 获取标准差
            if np.isnan(ref_mean) or np.isnan(ref_std):
                reference_texts.append(f"{dataset}: N/A")
            else:
                reference_texts.append(f"{dataset}: {ref_mean:.2f}±{ref_std:.2f}")  # mean±std格式

        # X轴刻度与标签（与现有风格一致）
        ax.set_xticks(range(len(TOKEN_LENGTHS)))
        ax.set_xticklabels([f'{tk}tk' for tk in TOKEN_LENGTHS], fontsize=24)
        # ax.set_xlabel('Noise Injection Length', fontsize=24, fontweight='bold')

        # 纵轴范围固定
        ax.set_ylim(-1.25, 0.25)
        ax.set_yticks([-1.2, -0.8, -0.4, 0.0])  # 只标记重点刻度
        ax.set_yticklabels([f'{v:.1f}' for v in ax.get_yticks()], fontsize=28)
        # ax.set_ylabel('Loss Difference', fontsize=24, fontweight='bold')
        # ax.grid(True, alpha=0.3)

        # 图例统一 右下角（不同噪音=不同数据集线颜色的标注）
        # ax.legend(fontsize=20, loc='lower right')

        # # 左下角显示reference（白底灰边）
        # ref_text = '\n'.join(reference_texts)
        # ax.text(
        #     0.02, 0.04, ref_text, transform=ax.transAxes,
        #     fontsize=20, verticalalignment='bottom', horizontalalignment='left',
        #     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
        # )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    # 保存为透明背景的PNG图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True, facecolor='none')
    plt.close()
    print(f"Combined line plot saved to: {output_path}")

def main():
    base_dir = '/home/evev/asap-dataset/+Loss'
    output_dir = '/home/evev/asap-dataset/+Loss_Plot/Phase5_1/Figure2'
    
    # 执行Token Length负相关分析
    print("开始Token Length vs Noise Loss负相关分析...")
    correlation_results = perform_token_length_correlation_analysis(
        base_dir, MODELS, DATASETS, TOKEN_LENGTHS
    )
    
    # 原有的绘图功能（如果需要）
    # all_model_data = collect_all_models_data()
    data = collect_all_models_data()
    
    # 创建线图
    create_line_plot_combined_single(data, output_dir, suffix='white_std')

if __name__ == '__main__':
    main()