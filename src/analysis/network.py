"""基因共表达网络分析（图3）—— WGCNA 管线。

Pipeline:
    加载数据 → 高变基因过滤(2000~5000) → Pearson相关矩阵
    → 软阈值选择(scale-free topology fit, R²>0.8)
    → 邻接矩阵(|corr|^β) → TOM矩阵 → 层次聚类 → 动态剪切
    → 模块-性状关联(eigengene × CDR/Braak/CERAD/APOE)
    → Hub基因(kME) → 3D网络可视化.

参考: WGCNA (Weighted Gene Co-expression Network Analysis) — Zhang & Horvath, 2005.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

from ..utils.config import RANDOM_SEED

# 复用聚类模块的函数
from ..analysis.clustering import (
    load_expression_matrix,
    load_sample_metadata,
    clean_metadata_for_analysis,
    filter_high_variance_genes,
)


# ═══════════════════════════════════════════════════════════════════════════
# 0. 数据加载（直接复用 clustering 模块，不再重复定义）
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# 1. 相关矩阵
# ═══════════════════════════════════════════════════════════════════════════

def compute_correlation_matrix(
    expr: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """计算基因间成对 Pearson 相关系数矩阵。

    Parameters
    ----------
    expr : pd.DataFrame, shape (n_genes, n_samples)
        基因 × 样本 表达矩阵（建议先做高变基因过滤，2000~5000 个）。
    method : str
        'pearson', 'spearman' 或 'kendall'。

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_genes)
        相关系数对称矩阵，行列索引均为基因 ID。
    """
    return expr.T.corr(method=method)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 软阈值选择（WGCNA 核心第1步）
# ═══════════════════════════════════════════════════════════════════════════

def _scale_free_fit(
    adj: np.ndarray,
) -> float:
    """计算邻接矩阵的 scale-free topology fit R²。

    对 log(degree_distribution) ~ log(degree) 做线性回归，
    返回 R² 作为 scale-free 拟合优度。

    Parameters
    ----------
    adj : np.ndarray, shape (n, n)
        邻接矩阵（对称，对角线无关）。

    Returns
    -------
    float
        线性回归 R² 值。越接近 1 越符合 scale-free 拓扑。
    """
    # 计算连通度 = 行和（排除对角线）
    k = adj.sum(axis=1) - np.diag(adj)
    k = k[k > 0]
    if len(k) < 5:
        return 0.0

    # 度数分布
    from collections import Counter as _Counter
    counts = _Counter(k)
    unique_k = np.array(sorted(counts.keys()))
    freq = np.array([counts[v] for v in unique_k])

    # log-log 线性回归
    log_k = np.log10(unique_k)
    log_freq = np.log10(freq)

    # 手动计算 R²
    mask = np.isfinite(log_k) & np.isfinite(log_freq)
    if mask.sum() < 4:
        return 0.0

    x = log_k[mask]
    y = log_freq[mask]
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def pick_soft_threshold(
    corr_df: pd.DataFrame,
    power_range: list[int] | None = None,
    r2_cutoff: float = 0.8,
) -> tuple[int, pd.DataFrame]:
    """遍历候选 power β 值，选择使 scale-free R² > cutoff 的最小 β。

    WGCNA 标准做法：对每个 β ∈ [1, 20]，计算邻接矩阵
    adj = |corr|^β，然后评估 scale-free topology fit R²。
    推荐选择 R² > 0.8 的最小 β（通常为 4~8）。

    Parameters
    ----------
    corr_df : pd.DataFrame, shape (n_genes, n_genes)
        基因间相关系数矩阵。
    power_range : list of int, optional
        候选 power 值列表，默认 [1, 2, ..., 20]。
    r2_cutoff : float
        R² 阈值，默认 0.8。

    Returns
    -------
    optimal_beta : int
        满足 R² > cutoff 的最小 power 值。若都不满足则返回 R² 最高的。
    sft_df : pd.DataFrame
        每行一个 power，列: ['power', 'r2', 'passed']。
    """
    if power_range is None:
        power_range = list(range(1, 21))

    corr_abs = np.abs(corr_df.values)
    # 对角线置 0，避免自环影响
    np.fill_diagonal(corr_abs, 0.0)

    records = []
    for beta in power_range:
        adj = corr_abs ** beta
        r2 = _scale_free_fit(adj)
        records.append({
            "power": beta,
            "r2": round(r2, 4),
            "passed": r2 > r2_cutoff,
        })

    sft_df = pd.DataFrame(records)

    # 选 R² > cutoff 的最小 power
    passed = sft_df[sft_df["passed"]]
    if len(passed) > 0:
        optimal_beta = int(passed["power"].min())
    else:
        # 都没过就选 R² 最高的
        optimal_beta = int(sft_df.loc[sft_df["r2"].idxmax(), "power"])

    return optimal_beta, sft_df


def soft_threshold_adjacency(
    corr_df: pd.DataFrame,
    beta: int,
) -> pd.DataFrame:
    """软阈值邻接矩阵：adj_{ij} = |corr_{ij}|^β。

    Parameters
    ----------
    corr_df : pd.DataFrame, shape (n_genes, n_genes)
        相关系数矩阵。
    beta : int
        软阈值 power 值。

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_genes)
        加权邻接矩阵（对称，对角线为 0）。
    """
    adj = np.abs(corr_df.values) ** beta
    np.fill_diagonal(adj, 0.0)
    return pd.DataFrame(adj, index=corr_df.index, columns=corr_df.columns)


# ═══════════════════════════════════════════════════════════════════════════
# 3. TOM（Topological Overlap Matrix，WGCNA 核心第2步）
# ═══════════════════════════════════════════════════════════════════════════

def compute_tom(adj_df: pd.DataFrame) -> pd.DataFrame:
    """计算 Topological Overlap Matrix。

    TOM_{ij} = (a_{ij} + Σ_k a_{ik}·a_{kj}) / (min(k_i, k_j) + 1 - a_{ij})

    其中 a_{ij} 是邻接矩阵元素，k_i = Σ_u a_{iu} 是节点 i 的连通度。
    TOM 衡量的是两个基因在共表达网络中的"拓扑重叠程度"——即它们
    不仅自身共表达，还与同一批其他基因共表达。

    Parameters
    ----------
    adj_df : pd.DataFrame, shape (n_genes, n_genes)
        软阈值邻接矩阵。

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_genes)
        TOM 相似性矩阵（对称）。
    """
    A = adj_df.values.copy()
    n = A.shape[0]

    # 连通度向量 k_i
    k = A.sum(axis=1)  # shape (n,)

    # TOM 分子: A + A·A
    # (A·A)_{ij} = Σ_k A_{ik}·A_{kj}
    numerator = A + A @ A  # shape (n, n)

    # 分母: min(k_i, k_j) + 1 - A_{ij}
    # 用广播: min(k_i, k_j) = outer min
    k_min = np.minimum(k[:, None], k[None, :])  # shape (n, n)
    denominator = k_min + 1.0 - A

    # 避免除以 0
    denominator = np.maximum(denominator, 1e-10)

    TOM = numerator / denominator

    # 对角线不参与后续聚类（会被忽略）
    np.fill_diagonal(TOM, 1.0)

    return pd.DataFrame(TOM, index=adj_df.index, columns=adj_df.columns)


# ═══════════════════════════════════════════════════════════════════════════
# 4. 层次聚类 + 动态剪切（WGCNA 核心第3步）
# ═══════════════════════════════════════════════════════════════════════════

def hierarchical_cluster_modules(
    tom_df: pd.DataFrame,
    min_module_size: int = 30,
    deep_split: int = 2,
) -> dict:
    """基于 TOM 差异度的层次聚类 + 动态剪切识别模块。

    1. 距离 = 1 - TOM
    2. Ward 层次聚类
    3. 按高度剪切树，合并过小模块到最近大模块

    Parameters
    ----------
    tom_df : pd.DataFrame, shape (n_genes, n_genes)
        TOM 相似性矩阵。
    min_module_size : int
        最小模块基因数。小于此值的模块被合并。
    deep_split : int
        剪切深度参数，值越大模块越多越细。2~4 常用。

    Returns
    -------
    partition : dict
        {gene_name: module_label}，module_label 为字符串如 "M0", "M1" …
        "M_grey" 为未分配基因。
    """
    gene_names = tom_df.index.tolist()
    n_genes = len(gene_names)

    # 距离 = 1 - TOM 相似度
    dist = 1.0 - tom_df.values
    # 只取上三角（scipy linkage 要求 condensed 格式）
    dist_condensed = squareform(dist, checks=False)

    # Ward 层次聚类
    Z = linkage(dist_condensed, method="ward")

    # 动态剪切：尝试不同高度寻找合适切割
    # deep_split 控制初始切割高度 = 最大高度的比例
    max_height = Z[-1, 2]
    cut_height = max_height * (0.35 / deep_split)

    # 第一次切割
    raw_labels = fcluster(Z, t=cut_height, criterion="distance")

    # 合并小模块（< min_module_size）到最近的大模块
    unique, counts = np.unique(raw_labels, return_counts=True)
    small_modules = set(unique[counts < min_module_size])

    if len(small_modules) > 0 and len(unique) > 1:
        # 为大模块计算中心（在 TOM 空间中的平均位置）
        large_modules = {m for m in unique if m not in small_modules}
        # 将小模块的每个基因重新分配给最近的大模块中心
        final_labels = raw_labels.copy()

        for module_id in small_modules:
            mask = raw_labels == module_id
            small_genes_idx = np.where(mask)[0]

            if len(large_modules) == 0:
                # 全部都是小模块，保留最大标签
                final_labels[mask] = max(unique, key=lambda x: counts[unique.tolist().index(x)])
                continue

            for idx in small_genes_idx:
                best_mod = None
                best_dist = float("inf")
                for big_mod in large_modules:
                    big_mask = raw_labels == big_mod
                    avg_dist = dist[idx, big_mask].mean()
                    if avg_dist < best_dist:
                        best_dist = avg_dist
                        best_mod = big_mod
                final_labels[idx] = best_mod

        raw_labels = final_labels

    # 重新编号模块为 M0, M1, …（按大小降序），小残余归入 M_grey
    counts_after = Counter(raw_labels.tolist())
    mod_sorted = sorted(counts_after.items(), key=lambda x: x[1], reverse=True)

    label_map: dict[int, str] = {}
    for rank, (mod_id, count) in enumerate(mod_sorted):
        if count < min_module_size:
            label_map[mod_id] = "M_grey"
        else:
            label_map[mod_id] = f"M{rank}"

    # 清理 grey 编号冲突
    used_names = set(label_map.values())
    for mod_id in mod_sorted:
        mid = mod_id[0]
        lbl = label_map[mid]
        if lbl in used_names:
            continue

    partition = {gene: label_map[lab] for gene, lab in zip(gene_names, raw_labels)}

    return partition


# ═══════════════════════════════════════════════════════════════════════════
# 5. 模块统计
# ═══════════════════════════════════════════════════════════════════════════

def module_summary(partition: dict) -> pd.DataFrame:
    """统计各模块基因数量，按大小降序排列。

    Parameters
    ----------
    partition : dict
        {gene_name: module_label}。

    Returns
    -------
    pd.DataFrame
        列: ['module', 'n_genes', 'pct']。
    """
    counter = Counter(partition.values())
    total = sum(counter.values())

    records = []
    for mod_label, count in counter.most_common():
        records.append({
            "module": mod_label,
            "n_genes": count,
            "pct": round(count / total * 100, 1),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 6. 模块 eigengene
# ═══════════════════════════════════════════════════════════════════════════

def module_eigengene(
    expr: pd.DataFrame,
    partition: dict,
) -> pd.DataFrame:
    """计算每个模块的 eigengene（第一主成分），即模块代表表达值。

    Parameters
    ----------
    expr : pd.DataFrame, shape (n_genes, n_samples)
        基因 × 样本 表达矩阵。
    partition : dict
        {gene_name: module_label}。

    Returns
    -------
    pd.DataFrame, shape (n_samples, n_modules)
        每列一个模块的 eigengene，行索引为 sample_id。
        不包括 M_grey 模块。
    """
    from sklearn.decomposition import PCA

    # 按模块分组基因（跳过 M_grey）
    module_genes: dict[str, list[str]] = {}
    for gene, mod in partition.items():
        if mod == "M_grey" or gene not in expr.index:
            continue
        module_genes.setdefault(mod, []).append(gene)

    columns: dict[str, np.ndarray] = {}
    for mod_id, genes in sorted(module_genes.items()):
        sub = expr.loc[genes].T  # samples × genes_of_this_module
        if sub.shape[1] == 1:
            columns[mod_id] = sub.iloc[:, 0].values
        else:
            pc1 = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(sub.values)
            columns[mod_id] = pc1[:, 0]

    return pd.DataFrame(columns, index=expr.columns)


# ═══════════════════════════════════════════════════════════════════════════
# 7. 模块-临床关联
# ═══════════════════════════════════════════════════════════════════════════

def module_clinical_corr(
    module_expr: pd.DataFrame,
    metadata: pd.DataFrame,
    clinical_cols: list[str] | None = None,
) -> pd.DataFrame:
    """计算各模块 eigengene 与临床指标的 Pearson r。

    Parameters
    ----------
    module_expr : pd.DataFrame, shape (n_samples, n_modules)
        module_eigengene 的输出。
    metadata : pd.DataFrame
        样本临床元数据（索引为 sample_id）。
    clinical_cols : list of str, optional
        默认 ['CDR', 'Braak', 'CERAD', 'ageDeath']。

    Returns
    -------
    pd.DataFrame, shape (n_modules, n_clinical_cols)
        相关性矩阵，值为 Pearson r。
    """
    if clinical_cols is None:
        clinical_cols = ["CDR", "Braak", "CERAD", "ageDeath"]

    common = module_expr.index.intersection(metadata.index)

    data: dict[str, dict[str, float]] = {}
    for mod_col in module_expr.columns:
        data[mod_col] = {}
        for clin_col in clinical_cols:
            x = module_expr.loc[common, mod_col]
            y = pd.to_numeric(metadata.loc[common, clin_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 3:
                data[mod_col][clin_col] = np.nan
            else:
                r, _ = pearsonr(x[mask], y[mask])
                data[mod_col][clin_col] = round(r, 4)

    return pd.DataFrame(data).T


# ═══════════════════════════════════════════════════════════════════════════
# 8. Hub 基因（kME — Module Membership）
# ═══════════════════════════════════════════════════════════════════════════

def get_hub_genes_kme(
    expr: pd.DataFrame,
    partition: dict,
    module_expr: pd.DataFrame | None = None,
    top_n: int = 10,
) -> dict[str, list[tuple[str, float]]]:
    """按 kME（Module Membership）识别各模块枢纽基因。

    kME = |corr(gene_expression, module_eigengene)|
    值越接近 1，该基因越能代表这个模块。

    Parameters
    ----------
    expr : pd.DataFrame, shape (n_genes, n_samples)
        基因 × 样本 表达矩阵（高变基因子集）。
    partition : dict
        {gene_name: module_label}。
    module_expr : pd.DataFrame, optional
        module_eigengene 的输出。若为 None 则内部计算。
    top_n : int
        每个模块选前 N 个 hub 基因。

    Returns
    -------
    dict
        {module_label: [(gene_name, kME), ...]}，按 kME 降序排列。
        不包括 M_grey。
    """
    if module_expr is None:
        module_expr = module_eigengene(expr, partition)

    hubs: dict[str, list[tuple[str, float]]] = {}

    for mod_label in module_expr.columns:
        if mod_label == "M_grey":
            continue

        me = module_expr[mod_label]
        kme_list = []

        for gene, mod in partition.items():
            if mod != mod_label or gene not in expr.index:
                continue
            gene_expr = expr.loc[gene]
            common = gene_expr.index.intersection(me.index)
            if len(common) < 3:
                continue
            r, _ = pearsonr(gene_expr[common], me[common])
            kme_list.append((gene, abs(r)))

        kme_list.sort(key=lambda x: x[1], reverse=True)
        hubs[mod_label] = kme_list[:top_n]

    return hubs
