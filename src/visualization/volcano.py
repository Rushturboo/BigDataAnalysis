"""火山图 — 差异表达基因可视化（图2）。

可视化编码:
- X 轴 → log2(Fold Change)，反映差异倍数
- Y 轴 → -log10(Adjusted P-value)，反映统计显著性
- 颜色 → 显著上调 (红色)、显著下调 (蓝色)、不显著 (灰色)
- 标记形状/特殊标注 → 已知 AD 关键基因 (星形金点)
- 悬停 → 基因名、log2FC、原始 p 值、FDR (padj)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

KNOWN_AD_GENES = ["APP", "PSEN1", "PSEN2", "MAPT", "APOE", "TREM2"]


def plot_volcano(
    de_results: pd.DataFrame,
    *,
    p_col: str = "padj",
    p_threshold: float = 0.01,
    fc_threshold: float = 1.0,
    top_n: int = 10,
    highlight_genes: list[str] | None = None,
    title: str = "Volcano Plot",
) -> go.Figure:
    """绘制单个差异表达火山图，支持交互式悬停与关键基因高亮。

    Parameters
    ----------
    de_results : pd.DataFrame
        差异表达分析结果。必须包含 'gene' 和 'log2FC' 列，以及由 p_col 指定的列。
    p_col : str, default 'padj'
        用于判断显著性的列名（通常为 padj 或 pvalue）。
    p_threshold : float, default 0.01
        显著性 P 值的阈值。
    fc_threshold : float, default 1.0
        log2 Fold Change 的绝对值阈值。
    top_n : int, default 10
        自动标注 log2FC 差异最大的前 N 个基因。
    highlight_genes : list of str, optional
        额外需要高亮显示的基因 Symbol 列表。
    title : str, default 'Volcano Plot'
        图表标题。

    Returns
    -------
    fig : plotly.graph_objects.Figure
        交互式火山图对象。
    """

    if de_results is None or de_results.empty:
        raise ValueError("de_results must be a non-empty DataFrame")

    df = de_results.copy()
    if "gene" not in df.columns or "log2FC" not in df.columns:
        raise ValueError("de_results must include 'gene' and 'log2FC' columns")

    df = df.reset_index(drop=True)
    df["p_for_plot"] = df[p_col].replace(0, np.nan)
    df["neg_log10_p"] = -np.log10(df["p_for_plot"])
    df["abs_log2FC"] = np.abs(df["log2FC"])

    significant_mask = df["p_for_plot"].fillna(1) < p_threshold
    up_mask = (df["log2FC"] >= fc_threshold) & significant_mask
    down_mask = (df["log2FC"] <= -fc_threshold) & significant_mask
    neutral_mask = ~(up_mask | down_mask)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["log2FC"],
            y=df["neg_log10_p"],
            mode="markers",
            marker=dict(
                size=8,
                color=np.where(up_mask, "#d62728", np.where(down_mask, "#1f77b4", "#7f7f7f")),
                opacity=0.8,
            ),
            text=df["gene"],
            hovertemplate=(
                "Gene: %{text}<br>"
                "log2FC: %{x:.3f}<br>"
                "-log10(p): %{y:.3f}<br>"
                "p-value: %{customdata[0]:.3e}<br>"
                "FDR: %{customdata[1]:.3e}<extra></extra>"
            ),
            customdata=df[["pvalue", "padj"]].values,
            showlegend=False,
        )
    )

    highlight_names = set(highlight_genes or []) | set(KNOWN_AD_GENES)
    if "gene_symbol" in df.columns:
        highlight_mask = df["gene_symbol"].isin(highlight_names)
    else:
        highlight_mask = df["gene"].isin(highlight_names)
    highlight_df = df[highlight_mask] 
    if not highlight_df.empty:
        fig.add_trace(
            go.Scatter(
                x=highlight_df["log2FC"],
                y=highlight_df["neg_log10_p"],
                mode="markers+text",
                text=highlight_df["gene"],
                textposition="top center",
                marker=dict(size=13, color="#ff7f0e", symbol="star-diamond", line=dict(width=1, color="black")),
                hovertemplate=(
                    "Gene: %{text}<br>"
                    "log2FC: %{x:.3f}<br>"
                    "-log10(p): %{y:.3f}<extra></extra>"
                ),
                name="AD Genes",
                showlegend=True,
            )
        )

    label_candidates = df[df["significant"]].copy() if "significant" in df.columns else df.copy()
    if label_candidates.empty:
        label_candidates = df.copy()

    label_candidates = pd.concat(
        [
            label_candidates.nlargest(top_n, "abs_log2FC"),
            highlight_df,
        ]
    ).drop_duplicates(subset="gene")

    label_df = label_candidates.head(top_n + len(highlight_df))
    fig.add_trace(
        go.Scatter(
            x=label_df["log2FC"],
            y=label_df["neg_log10_p"],
            mode="markers+text",
            text=label_df["gene"],
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(color="black", size=10),
            hovertemplate=(
                "Gene: %{text}<br>"
                "log2FC: %{x:.3f}<br>"
                "-log10(p): %{y:.3f}<extra></extra>"
            ),
            name="Top 10 Genes",
            showlegend=True,
        )
    )

    fig.add_vline(x=fc_threshold, line_dash="dash", line_color="black", line_width=1)
    fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="black", line_width=1)
    fig.add_hline(y=-np.log10(p_threshold), line_dash="dash", line_color="black", line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="log2FC",
        yaxis_title="-log10(p)",
        template="plotly_white",
        height=600,
        width=800,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_volcano_matrix(
    comparisons: dict[str, pd.DataFrame],
    *,
    p_col: str = "padj",
    p_threshold: float = 0.01,
    fc_threshold: float = 1.0,
    top_n: int = 8,
    title: str = "Volcano Plot Matrix",
    n_cols: int = 3,
) -> go.Figure:
    """将多个差异表达火山图并排展示为子图矩阵。

    Parameters
    ----------
    comparisons : dict of {str : pd.DataFrame}
        字典键为对比名称（如 'Cluster 0 vs 1'），值为对应的差异分析 DataFrame。
    p_col : str, default 'padj'
        用于显著性的列名。
    p_threshold : float, default 0.01
        显著性 P 值阈值。
    fc_threshold : float, default 1.0
        log2FC 阈值。
    top_n : int, default 8
        每个子图中自动标注的前 N 个基因数。
    title : str, default 'Volcano Plot Matrix'
        总图标题。
    n_cols : int, default 3
        子图矩阵的列数。

    Returns
    -------
    fig : plotly.graph_objects.Figure
        并排展示的子图矩阵。
    """
    
    names = list(comparisons.keys())
    n_rows = int(np.ceil(len(names) / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=names,
        horizontal_spacing=0.08,
        vertical_spacing=0.16,
    )

    for idx, name in enumerate(names):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        single_fig = plot_volcano(
            comparisons[name],
            p_col=p_col,
            p_threshold=p_threshold,
            fc_threshold=fc_threshold,
            top_n=top_n,
            title=name,
        )
        for trace in single_fig.data:
            if trace.name in ["AD Genes", "Top 10 Genes"]:
                trace.mode = "markers" 
                trace.text = None      
            fig.add_trace(trace, row=row, col=col)

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=250 * n_rows,
        width=320 * n_cols,
        showlegend=False,
    )
    return fig
