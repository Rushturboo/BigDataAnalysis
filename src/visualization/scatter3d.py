"""3D 聚类散点图 — Plotly 交互式（图1）。

可视化编码:
- 颜色 → 聚类标签 (Cluster / 亚型)
- 形状 → CDR 临床痴呆评分 (0-5)
- 大小 → Braak 神经纤维缠结分期 (0-6)
- 悬停 → individualID, ageDeath, APOE 基因型, CERAD
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# CDR 各水平的标记符号（Plotly 3D 支持的符号名）
_CDR_SYMBOL_MAP: dict[str, str] = {
    "0.0": "circle",
    "0.5": "circle-open",
    "1.0": "diamond",
    "2.0": "diamond-open",
    "3.0": "square",
    "4.0": "square-open",
    "5.0": "cross",
}

_DEFAULT_HOVER_COLS: list[str] = [
    "individualID",
    "ageDeath",
    "apoeGenotype_str",
    "CERAD",
    "sex",
]


def plot_3d_clusters(
    umap_emb: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    symbol_col: str = "CDR",
    size_col: str = "Braak",
    hover_data: list[str] | None = None,
    title: str = "AD Subtypes — 3D UMAP Clustering (BM_36 Parahippocampal Gyrus)",
    color_palette: list[str] | None = None,
    marker_size: int = 5,
    marker_opacity: float = 0.8,
) -> go.Figure:
    """创建 AD 亚型 3D 交互式聚类散点图。

    Parameters
    ----------
    umap_emb : np.ndarray, shape (n_samples, 3)
        UMAP 3 维嵌入坐标。
    labels : np.ndarray, shape (n_samples,)
        整数聚类标签。
    metadata : pd.DataFrame, shape (n_samples, *)
        样本元数据，索引为 sample_id。必须包含 symbol_col, size_col
        以及 hover_data 中列出的所有列。
    symbol_col : str
        用于标记形状的元数据列名，默认 'CDR'。
    size_col : str
        用于标记大小的元数据列名，默认 'Braak'。
    hover_data : list of str, optional
        悬停提示中显示的列。默认包含 individualID, ageDeath,
        apoeGenotype_str, CERAD, sex。
    title : str
        图表标题。
    color_palette : list of str, optional
        聚类颜色调色板，默认 ``px.colors.qualitative.Bold``。
    marker_size : int
        基础标记大小。
    marker_opacity : float
        标记透明度 (0-1)。

    Returns
    -------
    fig : plotly.graph_objects.Figure
        可交互的 3D 散点图。
    """
    if hover_data is None:
        hover_data = _DEFAULT_HOVER_COLS

    if color_palette is None:
        color_palette = px.colors.qualitative.Bold

    # 构建 plot DataFrame
    plot_df = pd.DataFrame({
        "UMAP1": umap_emb[:, 0],
        "UMAP2": umap_emb[:, 1],
        "UMAP3": umap_emb[:, 2],
        "Cluster": labels.astype(str),
    }, index=metadata.index)

    # 合并元数据的可视化列
    plot_df[symbol_col] = metadata[symbol_col].astype(str)
    plot_df[size_col] = pd.to_numeric(metadata[size_col], errors="coerce").fillna(0)

    for col in hover_data:
        plot_df[col] = metadata[col].values

    # 确保 CDR 符号有对应映射（未知值回退到 circle）
    # 只保留数据中实际出现的 CDR 值对应的符号
    unique_cdr = sorted(plot_df[symbol_col].unique())
    symbol_seq = [_CDR_SYMBOL_MAP.get(v, "circle") for v in unique_cdr]

    fig = px.scatter_3d(
        plot_df,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color="Cluster",
        symbol=symbol_col,
        size=size_col,
        hover_data=hover_data,
        color_discrete_sequence=color_palette,
        symbol_sequence=symbol_seq,
        size_max=15,
        title=title,
        labels={"UMAP1": "UMAP1", "UMAP2": "UMAP2", "UMAP3": "UMAP3"},
    )

    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        scene=dict(
            xaxis_title="UMAP1",
            yaxis_title="UMAP2",
            zaxis_title="UMAP3",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )

    return fig


def plot_multi_view_comparison(
    umap_emb: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    marker_size: int = 4,
    marker_opacity: float = 0.8,
) -> go.Figure:
    """并排多视图对比：同一 3D UMAP 布局，分别按 Cluster / CERAD / Braak / APOE 着色。

    Parameters
    ----------
    umap_emb : np.ndarray, shape (n_samples, 3)
    labels : np.ndarray, shape (n_samples,)
    metadata : pd.DataFrame
        必须包含列: CERAD, Braak, apoeGenotype_str, individualID
    marker_size : int
    marker_opacity : float

    Returns
    -------
    fig : plotly.graph_objects.Figure
        2×2 子图布局的 3D 散点图。
    """
    n_samples = len(umap_emb)
    cluster_str = pd.Series(labels.astype(str))

    # ---- 准备各视图的颜色列 ----
    cerad_str = metadata["CERAD"].fillna(0).astype(int).astype(str)
    braak_str = metadata["Braak"].fillna(0).astype(int).astype(str)
    apoe_str = metadata["apoeGenotype_str"].fillna("Unknown")

    hover_text = [
        f"ID: {row.individualID}<br>Age: {row.ageDeath:.0f}<br>"
        f"CDR: {row.CDR}<br>CERAD: {int(row.CERAD)}<br>Braak: {int(row.Braak)}<br>"
        f"APOE: {row.apoeGenotype_str}"
        for _, row in metadata.iterrows()
    ]

    # ---- 构建 4 个视图的定义 ----
    views = [
        ("By Cluster", cluster_str, px.colors.qualitative.Bold),
        ("By CERAD (Neuritic Plaque)", cerad_str, px.colors.sequential.Reds[1:]),
        ("By Braak (NFT Stage)", braak_str, px.colors.sequential.Oranges[1:]),
        ("By APOE Genotype", apoe_str, px.colors.qualitative.Set2),
    ]

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}],
               [{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[v[0] for v in views],
        horizontal_spacing=0.02,
        vertical_spacing=0.02,
    )

    for idx, (title, color_series, palette) in enumerate(views):
        row = idx // 2 + 1
        col = idx % 2 + 1

        unique_vals = sorted(color_series.unique(), key=lambda x: (x.isdigit(), x))
        for vi, val in enumerate(unique_vals):
            mask = (color_series == val).values
            color = palette[vi % len(palette)]

            fig.add_trace(
                go.Scatter3d(
                    x=umap_emb[mask, 0],
                    y=umap_emb[mask, 1],
                    z=umap_emb[mask, 2],
                    mode="markers",
                    name=str(val),
                    legendgroup=title,
                    legendgrouptitle_text=title if vi == 0 else None,
                    showlegend=True,
                    marker=dict(size=marker_size, opacity=marker_opacity, color=color),
                    text=[hover_text[i] for i in range(n_samples) if mask[i]],
                    hoverinfo="text",
                ),
                row=row, col=col,
            )

        fig.update_scenes(
            xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3",
            row=row, col=col,
        )

    fig.update_layout(
        title="Multi-View Comparison: Same 3D UMAP, Different Clinical Annotations",
        title_font_size=16,
        margin=dict(l=0, r=0, b=0, t=60),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            groupclick="toggleitem",
        ),
    )

    return fig
