"""差异表达分析：组间统计检验与倍数变化计算（图2）。

主要功能:
- 计算 log2 Fold Change (log2FC)
- 执行独立样本 t 检验 (Student's or Welch's t-test)
- 执行 Benjamini-Hochberg (FDR) 多重检验校正
- 生成适用于火山图可视化的标准差异分析结果表
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _benjamini_hochberg(pvalues: np.ndarray | list[float]) -> np.ndarray:
    """对一维 p-value 数组执行 Benjamini-Hochberg FDR 校正。

    Parameters
    ----------
    pvalues : np.ndarray | list[float]
        原始 p 值数组。

    Returns
    -------
    np.ndarray
        校正后的 q 值（FDR）数组，保持原始顺序，包含 NaN 处理。
    """

    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    valid_mask = ~np.isnan(p)
    if not np.any(valid_mask):
        return q

    valid_p = p[valid_mask]
    order = np.argsort(valid_p)
    sorted_p = valid_p[order]
    ranks = np.arange(1, len(sorted_p) + 1)

    adjusted = sorted_p * len(sorted_p) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid_mask] = q_valid
    return q


def _normalize_labels(labels, index: pd.Index) -> pd.Series:
    """将输入的 labels 转换为与表达矩阵样本名对齐的 pandas Series。

    Parameters
    ----------
    labels : any
        样本分组信息（Series, list, or array）。
    index : pd.Index
        表达矩阵的列索引（样本 ID）。

    Returns
    -------
    pd.Series
        与表达矩阵索引完全对齐的分组信息。

    Raises
    ------
    ValueError
        当标签长度不匹配或索引无法对齐时抛出。
    """

    if isinstance(labels, pd.Series):
        label_series = labels.copy()
        if len(label_series) != len(index):
            if label_series.index.is_unique and label_series.index.intersection(index).shape[0] == len(index):
                label_series = label_series.reindex(index)
            else:
                raise ValueError("labels length does not match expression matrix columns")
        return label_series

    if len(labels) != len(index):
        raise ValueError("labels length does not match expression matrix columns")

    return pd.Series(labels, index=index)


def _label_matches(value, target) -> bool:
    """内部匹配函数：兼容字符串和数字类型的标签比较。

    Parameters
    ----------
    value : any
        待检测的样本标签。
    target : any
        目标对比组标签。

    Returns
    -------
    bool
        标签是否匹配（忽略空格与类型差异）。
    """

    if pd.isna(value):
        return False
    return str(value).strip() == str(target).strip()


def calc_differential_expression(
    expr: pd.DataFrame,
    labels,
    group_a,
    group_b,
    *,
    equal_var: bool = False,
    min_group_size: int = 2,
    method: str = "ttest",
) -> pd.DataFrame:
    """计算两组间的差异表达统计量。

    计算过程包括均值计算、log2FC、统计检验及 FDR 校正。

    Parameters
    ----------
    expr : pd.DataFrame
        基因表达矩阵（genes × samples）。数据应预先经过标准化（如 log2CPM）。
    labels : any
        样本的分组信息。
    group_a : str | int
        对比组 A（对照组）的标签名。
    group_b : str | int
        对比组 B（实验组）的标签名。计算为 B - A。
    equal_var : bool, default False
        t 检验是否假设等方差。False 则执行 Welch's t-test。
    min_group_size : int, default 2
        执行统计检验所需的最小有效样本数。
    method : str, default 'ttest'
        检验方法：'ttest' (参数检验) 或 'wilcoxon' (非参数检验，Mann-Whitney U)。

    Returns
    -------
    pd.DataFrame
        差异分析结果表，包含列:
        - 'gene': 基因 ID
        - 'log2FC': 组间平均值差异 (MeanB - MeanA)
        - 'pvalue': 统计检验 p 值
        - 'padj': BH 校正后的 FDR
        - 'significant': 基于默认阈值 (FDR<0.05 & |log2FC|>1) 的显著性判定
        - 其他辅助列: mean_A, mean_B, neg_log10_p 等。
    """

    expr_df = expr.copy()
    if not isinstance(expr_df, pd.DataFrame):
        raise TypeError("expr must be a pandas DataFrame")

    if expr_df.shape[1] == 0:
        raise ValueError("expression matrix has no samples")

    label_series = _normalize_labels(labels, expr_df.columns)
    mask_a = label_series.apply(lambda x: _label_matches(x, group_a))
    mask_b = label_series.apply(lambda x: _label_matches(x, group_b))

    if mask_a.sum() < min_group_size or mask_b.sum() < min_group_size:
        raise ValueError("each group must contain at least two samples")

    rows: list[dict[str, float | int | str]] = []
    for gene in expr_df.index:
        values_a = expr_df.loc[gene, mask_a].astype(float).to_numpy()
        values_b = expr_df.loc[gene, mask_b].astype(float).to_numpy()

        values_a = values_a[~np.isnan(values_a)]
        values_b = values_b[~np.isnan(values_b)]

        if len(values_a) < min_group_size or len(values_b) < min_group_size:
            rows.append(
                {
                    "gene": gene,
                    "log2FC": np.nan,
                    "pvalue": np.nan,
                    "padj": np.nan,
                    "mean_A": np.nan,
                    "mean_B": np.nan,
                    "n_A": int(mask_a.sum()),
                    "n_B": int(mask_b.sum()),
                    "t_statistic": np.nan,
                    "abs_log2FC": np.nan,
                    "direction": "ns",
                }
            )
            continue

        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        log2fc = float(mean_b - mean_a)

        try:
            if method.lower() == "ttest":
                test_stat, pvalue = stats.ttest_ind(values_a, values_b, equal_var=equal_var, nan_policy="omit")
            elif method.lower() == "wilcoxon":
                test_stat, pvalue = stats.mannwhitneyu(values_a, values_b, alternative="two-sided", nan_policy="omit")
            else:
                raise ValueError(f"unsupported method: {method}")
        except Exception:
            test_stat, pvalue = np.nan, np.nan

        rows.append(
            {
                "gene": gene,
                "log2FC": log2fc,
                "pvalue": float(pvalue) if not np.isnan(pvalue) else np.nan,
                "padj": np.nan,
                "mean_A": mean_a,
                "mean_B": mean_b,
                "n_A": int(len(values_a)),
                "n_B": int(len(values_b)),
                "test_statistic": float(test_stat) if not np.isnan(test_stat) else np.nan,
                "abs_log2FC": abs(log2fc),
                "direction": "up" if log2fc > 0 else "down" if log2fc < 0 else "ns",
            }
        )

    results = pd.DataFrame(rows)
    if not results.empty and "pvalue" in results.columns:
        results["padj"] = _benjamini_hochberg(results["pvalue"].to_numpy())
        results["neg_log10_p"] = -np.log10(results["pvalue"].replace(0, np.nan))
        results["neg_log10_padj"] = -np.log10(results["padj"].replace(0, np.nan))
        results["significant"] = (results["padj"] < 0.05) & (results["abs_log2FC"] > 1.0)

    return results


def pairwise_de_all(expr: pd.DataFrame, labels) -> dict[str, pd.DataFrame]:
    """对分组信息中所有的两两组合生成穷举式差异表达分析。

    Parameters
    ----------
    expr : pd.DataFrame
        表达矩阵。
    labels : any
        样本的分组标签。

    Returns
    -------
    dict[str, pd.DataFrame]
        键名为对比组合（如 'A_vs_B'），键值为对应的差异分析结果 DataFrame。
    """
    
    label_series = _normalize_labels(labels, expr.columns)
    groups = sorted({str(x) for x in label_series.unique()})
    results: dict[str, pd.DataFrame] = {}

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            group_a = groups[i]
            group_b = groups[j]
            name = f"{group_a}_vs_{group_b}"
            results[name] = calc_differential_expression(expr, label_series, group_a, group_b)

    return results
