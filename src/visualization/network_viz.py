"""基因共表达网络 3D 可视化 — WGCNA 管线（图3）。

可视化编码:
- 节点颜色 → WGCNA 模块归属
- 节点大小 → kME（Module Membership），越大的越核心
- 边透明度 → 邻接权重（|corr|^β）
- 悬停 → 基因名、模块号、kME 值
- 标注 → top N hub 基因名
"""

from __future__ import annotations

import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px


def build_viz_graph(
    adj_df: "pd.DataFrame",
    top_fraction: float = 0.05,
) -> nx.Graph:
    """从软阈值邻接矩阵构建用于可视化的 NetworkX 图。

    保留权重 top_fraction 的边（默认 5%），控制 3D 渲染复杂度。

    Parameters
    ----------
    adj_df : pd.DataFrame, shape (n_genes, n_genes)
        软阈值邻接矩阵 (|corr|^β)。
    top_fraction : float
        保留的边比例，默认 0.05（5%）。

    Returns
    -------
    nx.Graph
        节点属性: 'name'=基因ID, 'weight'=节点总连通度。
        边属性: 'weight'=邻接权重。
    """
    values = adj_df.values.copy()
    n = values.shape[0]

    # 取上三角权重，设阈值
    triu_vals = values[np.triu_indices(n, k=1)]
    if len(triu_vals) == 0:
        return nx.Graph()

    cutoff = np.percentile(triu_vals[triu_vals > 0], (1 - top_fraction) * 100)

    G = nx.Graph()
    gene_names = adj_df.index.tolist()

    for i, gene in enumerate(gene_names):
        # 节点连通度 = 该基因所有权重之和
        G.add_node(i, name=gene, weight=float(values[i].sum()))

    # 只保留高于 cutoff 的边
    rows, cols = np.where(np.triu(values, k=1) > cutoff)
    for i, j in zip(rows, cols):
        G.add_edge(int(i), int(j), weight=float(values[i, j]))

    return G


def plot_network_3d(
    G: nx.Graph,
    partition: dict,
    kme_hubs: dict[str, list[tuple[str, float]]] | None = None,
    max_nodes: int = 500,
    layout_seed: int = 42,
    title: str = "Gene Co-expression Network — 3D WGCNA Modules (BM_36)",
) -> go.Figure:
    """绘制 3D 交互式基因共表达网络（WGCNA 管线）。

    力导向布局 → 节点按模块着色，大小 = kME，边的透明度 = 权重。

    Parameters
    ----------
    G : nx.Graph
        基因共表达网络（build_viz_graph 的输出）。
    partition : dict
        {gene_name: module_label}。
    kme_hubs : dict, optional
        {module_label: [(gene_name, kME), ...]}，用于标注 hub 和调整大小。
    max_nodes : int
        图中最多展示的节点数（取连通度最高的）。
    layout_seed : int
        spring_layout 随机种子。
    title : str
        图表标题。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # ── 缩减节点数（取 top 连通度）──
    if G.number_of_nodes() > max_nodes:
        top_nodes = sorted(
            G.nodes(data=True),
            key=lambda x: x[1].get("weight", 0),
            reverse=True,
        )[:max_nodes]
        G = G.subgraph([n for n, _ in top_nodes]).copy()

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # ── 构建 kME 查找 ──
    kme_lookup: dict[str, float] = {}
    if kme_hubs:
        for mod_label, genes in kme_hubs.items():
            for gene_name, kme in genes:
                kme_lookup[gene_name] = max(kme_lookup.get(gene_name, 0), kme)

    # ── 3D 布局 ──
    pos = nx.spring_layout(
        G, dim=3, seed=layout_seed, k=1.5, iterations=80, weight="weight",
    )

    # ── 节点属性 ──
    x_nodes = np.array([pos[n][0] for n in G.nodes()])
    y_nodes = np.array([pos[n][1] for n in G.nodes()])
    z_nodes = np.array([pos[n][2] for n in G.nodes()])

    gene_names = [G.nodes[n]["name"] for n in G.nodes()]
    module_labels = [partition.get(name, "M_grey") for name in gene_names]
    kme_vals = np.array([kme_lookup.get(name, 0.1) for name in gene_names])

    # ── 边坐标 ──
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])
        edge_z.extend([pos[u][2], pos[v][2], None])

    # ── 构建图形 ──
    fig = go.Figure()

    # 边 trace
    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode="lines",
        line=dict(color="rgba(150,150,150,0.12)", width=0.5),
        name=f"Edges ({n_edges:,})",
        hoverinfo="none",
    ))

    # 节点按模块分组绘制
    unique_mods = sorted(set(module_labels), key=lambda m: (
        0 if m != "M_grey" else 999,
        module_labels.count(m),
    ), reverse=True)

    palette = px.colors.qualitative.Bold
    if len(unique_mods) > len(palette):
        palette = palette * (len(unique_mods) // len(palette) + 1)

    grey_color = "rgba(180,180,180,0.5)"

    # 收集用于标注的 hub 基因名
    hub_names: set[str] = set()
    if kme_hubs:
        for genes in kme_hubs.values():
            for gene_name, _ in genes[:5]:  # top 5 per module
                hub_names.add(gene_name)

    for idx, mod in enumerate(unique_mods):
        mask = np.array([m == mod for m in module_labels])
        if not mask.any():
            continue

        mod_kme = kme_vals[mask]
        mod_names = np.array(gene_names)[mask]

        # 节点大小: 基础 3 + kME * 12
        sizes = 3 + mod_kme * 12

        hover = []
        for name, kme in zip(mod_names, mod_kme):
            is_hub = "★ " if name in hub_names else ""
            hover.append(
                f"{is_hub}Gene: {name}<br>Module: {mod}<br>kME: {kme:.4f}"
            )

        color = grey_color if mod == "M_grey" else palette[idx % len(palette)]

        fig.add_trace(go.Scatter3d(
            x=x_nodes[mask], y=y_nodes[mask], z=z_nodes[mask],
            mode="markers",
            name=f"{mod} ({mask.sum()} genes)",
            marker=dict(size=sizes.tolist(), color=color, opacity=0.85),
            text=hover,
            hoverinfo="text",
        ))

    # ── 标注 top hub ──
    if kme_hubs:
        hub_label_list = []
        for mod_label in sorted(kme_hubs.keys()):
            for gene_name, kme in kme_hubs[mod_label][:3]:  # top 3 per module
                hub_label_list.append(f"{mod_label if mod_label != 'M_grey' else 'grey'}: {gene_name}")

        fig.add_annotation(
            x=0.99, y=0.99, xref="paper", yref="paper",
            text="<b>Top Hub Genes</b><br>" + "<br>".join(hub_label_list),
            showarrow=False,
            font=dict(size=10, color="white"),
            bgcolor="rgba(30,30,30,0.85)",
            bordercolor="gray",
            borderwidth=1,
            align="left",
            xanchor="right",
        )

    # ── 布局 ──
    unique_mods_no_grey = [m for m in unique_mods if m != "M_grey"]
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{n_nodes} genes · {n_edges:,} edges · {len(unique_mods_no_grey)} modules (WGCNA)</sup>",
            font=dict(size=16),
        ),
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    return fig


def plot_soft_threshold_diagnostics(
    sft_df: "pd.DataFrame",
    optimal_beta: int,
) -> "go.Figure":
    """软阈值选择诊断图：展示各 power 的 scale-free R²。

    Parameters
    ----------
    sft_df : pd.DataFrame
        pick_soft_threshold 的输出。列: ['power', 'r2', 'passed']。
    optimal_beta : int
        选中的最优 power。

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    fig = go.Figure()

    # R² 线
    fig.add_trace(go.Scatter(
        x=sft_df["power"],
        y=sft_df["r2"],
        mode="lines+markers",
        marker=dict(
            size=10,
            color=["green" if p else "gray" for p in sft_df["passed"]],
        ),
        name="Scale-free R²",
    ))

    # R²=0.8 参考线
    fig.add_hline(
        y=0.8, line_dash="dash", line_color="red",
        annotation_text="R² = 0.8",
        annotation_position="bottom right",
    )

    # 标注最优 power
    optimal_r2 = sft_df.loc[sft_df["power"] == optimal_beta, "r2"].values[0]
    fig.add_trace(go.Scatter(
        x=[optimal_beta],
        y=[optimal_r2],
        mode="markers+text",
        marker=dict(size=16, color="red", symbol="star"),
        text=[f"β={optimal_beta}"],
        textposition="top center",
        name="Optimal β",
        showlegend=False,
    ))

    fig.update_layout(
        title=f"Scale-free Topology Fit — Optimal β = {optimal_beta} (R² = {optimal_r2:.3f})",
        xaxis=dict(title="Soft Threshold (power β)", dtick=1),
        yaxis=dict(title="Scale-free Topology Model Fit (R²)", range=[0, 1.05]),
        width=700,
        height=450,
    )

    return fig


def plot_module_clinical_heatmap(
    corr_df: "pd.DataFrame",
    title: str = "Module–Clinical Correlation Heatmap (WGCNA Modules)",
) -> "go.Figure":
    """绘制模块-临床相关性热图。

    Parameters
    ----------
    corr_df : pd.DataFrame, shape (n_modules, n_clinical_cols)
        module_clinical_corr 的输出。
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(corr_df.values, 3),
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Pearson r"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Clinical Variable",
        yaxis_title="Module",
        width=550,
        height=250 + 25 * len(corr_df),
    )

    return fig
