"""聚类分析：PCA → UMAP → K-means → 3D 散点图用（图1）。

Pipeline: 加载数据 → 高变基因过滤 → PCA(50维) → 确定最优K
         → K-means聚类(PCA空间) → UMAP(3维,仅可视化) → 3D散点图.

重要约束: K-means 在 PCA 降维后的 50 维空间上进行，
          UMAP 降维到 3 维仅用于可视化，不应作为聚类输入。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import silhouette_score
import umap

from ..utils.config import RANDOM_SEED

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_expression_matrix(path: str | Path) -> pd.DataFrame:
    """从 CSV 加载表达矩阵，使用 float32 减半内存占用。

    CSV 格式: 第 1 列 Gene (ENSG ID)，第 2+ 列为各样本的 log2(CPM+1) 值。
    即 ``genes × samples`` 布局。

    Parameters
    ----------
    path : str or Path
        logcpm 矩阵 CSV 文件路径。

    Returns
    -------
    pd.DataFrame, shape (n_genes, n_samples)
        行索引为基因 ENSG ID，列名为 sample_id，dtype=float32。
    """
    # Load as default float64 first (avoids dtype coercion issues during CSV parsing),
    # then downcast to float32 to halve memory.
    return pd.read_csv(path, index_col=0).astype(np.float32, copy=False)


def load_sample_metadata(path: str | Path) -> pd.DataFrame:
    """加载样本临床元数据 CSV。

    Parameters
    ----------
    path : str or Path
        样本元数据 CSV 文件路径。

    Returns
    -------
    pd.DataFrame, shape (n_samples, n_clinical_cols)
        行索引为 sample_id。
    """
    return pd.read_csv(path, index_col=0)


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------

# APOE 基因型数值 → 可读字符串映射
_APOE_MAP: dict[float, str] = {
    33.0: "ε3/ε3",
    34.0: "ε3/ε4",
    44.0: "ε4/ε4",
    23.0: "ε2/ε3",
    22.0: "ε2/ε2",
    24.0: "ε2/ε4",
}


def clean_metadata_for_analysis(meta: pd.DataFrame) -> pd.DataFrame:
    """清洗元数据，处理缺失值和异常格式。

    操作:
    1. ageDeath: 将 "90+" 字符串替换为 90.0，其余转为数值
    2. CDR / Braak / CERAD: 强制转为数值类型
    3. apoeGenotype_str: 新增可读列 (ε3/ε3 等)，缺失填 "Unknown"

    Parameters
    ----------
    meta : pd.DataFrame
        原始元数据 DataFrame，索引为 sample_id。

    Returns
    -------
    pd.DataFrame
        清洗后的元数据副本（不修改原数据）。
    """
    cleaned = meta.copy()

    # --- ageDeath: "90+" → 90.0 ---
    cleaned["ageDeath"] = (
        cleaned["ageDeath"]
        .astype(str)
        .str.strip()
        .replace("90+", "90.0")
        .pipe(pd.to_numeric, errors="coerce")
    )

    # --- 临床评分列强制数值 ---
    for col in ("CDR", "Braak", "CERAD"):
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # --- APOE 字符串列 ---
    cleaned["apoeGenotype_str"] = (
        cleaned["apoeGenotype"]
        .map(_APOE_MAP)
        .fillna("Unknown")
    )

    return cleaned


# ---------------------------------------------------------------------------
# 特征筛选
# ---------------------------------------------------------------------------

def filter_high_variance_genes(
    expr: pd.DataFrame,
    n_top: int = 3000,
    variance_threshold: float = 0.1,
) -> tuple[pd.DataFrame, list[str]]:
    """两步筛选高变基因。

    1. 从所有基因中按方差从大到小取前 ``n_top`` 个
    2. 再对筛选结果用 ``VarianceThreshold(threshold)`` 剔除近乎零方差的基因

    注意：先取 top N 再 VarianceThreshold，确保返回的基因数接近 n_top。

    Parameters
    ----------
    expr : pd.DataFrame, shape (n_genes, n_samples)
        基因 × 样本表达矩阵。
    n_top : int
        保留的高变基因数量。
    variance_threshold : float
        最小方差阈值，传递给 sklearn VarianceThreshold。

    Returns
    -------
    filtered_expr : pd.DataFrame, shape (n_selected, n_samples)
        筛选后的表达矩阵（仍为 genes × samples）。
    selected_genes : list of str
        被保留的基因 ID 列表。
    """
    # Step 1: 按方差降序取 top N 基因
    variances = expr.var(axis=1)
    top_genes = variances.nlargest(min(n_top, len(expr))).index.tolist()
    expr_top = expr.loc[top_genes]

    # Step 2: 用 VarianceThreshold 剔除其中近乎零方差的基因
    # sklearn VarianceThreshold 要求 samples × features
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(expr_top.T)

    surviving_mask = selector.get_support()
    final_genes = [g for g, keep in zip(top_genes, surviving_mask) if keep]

    return expr.loc[final_genes], final_genes


# ---------------------------------------------------------------------------
# 格式转换
# ---------------------------------------------------------------------------

def prepare_for_sklearn(expr: pd.DataFrame) -> np.ndarray:
    """转置表达矩阵：genes×samples → samples×genes，转为 float32。

    Parameters
    ----------
    expr : pd.DataFrame, shape (n_genes, n_samples)

    Returns
    -------
    np.ndarray, shape (n_samples, n_genes), dtype float32
    """
    return expr.T.values.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 降维
# ---------------------------------------------------------------------------

def run_pca(
    expr: np.ndarray,
    n_components: int = 50,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, PCA]:
    """对表达矩阵执行 PCA 降维。

    Parameters
    ----------
    expr : np.ndarray, shape (n_samples, n_features)
        方差过滤后的表达矩阵（samples × genes）。
    n_components : int
        保留的主成分数量。
    random_state : int

    Returns
    -------
    pca_result : np.ndarray, shape (n_samples, n_components)
        PCA 降维后的数据。
    pca_model : PCA
        已拟合的 PCA 模型（可查看 .explained_variance_ratio_ 等）。
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_result = pca.fit_transform(expr)
    return pca_result, pca


def run_umap(
    data: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, umap.UMAP]:
    """对 PCA 降维后的数据执行 UMAP 嵌入（仅用于 3D 可视化）。

    .. warning::
       UMAP 的 3 维嵌入仅用于可视化，不应作为 K-means 聚类的输入。
       聚类应始终在 PCA 空间（高维）上进行。

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        PCA 降维后的数据（如 215 × 50）。
    n_components : int
        嵌入维度（3 即 3D）。
    n_neighbors : int
        UMAP 邻居数。
    min_dist : float
        最小嵌入距离。
    random_state : int

    Returns
    -------
    embedding : np.ndarray, shape (n_samples, n_components)
        UMAP 嵌入坐标。
    reducer : umap.UMAP
        已拟合的 UMAP 对象。
    """
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(data)
    return embedding, reducer


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------

def run_kmeans(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = RANDOM_SEED,
    n_init: int = 10,
) -> tuple[np.ndarray, KMeans]:
    """对 PCA 降维数据执行 K-means 聚类。

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        PCA 降维后的数据。必须是 PCA 空间，不是 UMAP 嵌入。
    n_clusters : int
        簇数 K。
    random_state : int
    n_init : int
        不同初始化的运行次数（取最佳结果）。

    Returns
    -------
    labels : np.ndarray, shape (n_samples,), dtype int
        聚类标签（0 ~ n_clusters-1）。
    kmeans_model : KMeans
        已拟合的 KMeans 模型（可查看 .inertia_, .cluster_centers_）。
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = kmeans.fit_predict(data)
    return labels, kmeans


def determine_optimal_k(
    data: np.ndarray,
    k_range: range | list = range(2, 9),
    random_state: int = RANDOM_SEED,
) -> tuple[int, pd.DataFrame]:
    """肘部法则 + 轮廓系数确定最优 K。

    对 k_range 中的每个 K 值运行 KMeans，记录 inertia（肘部）和
    silhouette_score（轮廓系数），取轮廓系数最高的 K 作为推荐值。

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_features)
        PCA 降维后的数据。
    k_range : iterable
        待测试的 K 值范围，默认 range(2, 9) 即 2~8。
    random_state : int

    Returns
    -------
    optimal_k : int
        推荐的簇数。
    metrics_df : pd.DataFrame
        每行一个 K，列: ['k', 'inertia', 'silhouette_score']。
    """
    records: list[dict] = []

    for k in k_range:
        labels, km = run_kmeans(data, n_clusters=k, random_state=random_state)
        sil = silhouette_score(data, labels, random_state=random_state)
        records.append({
            "k": k,
            "inertia": km.inertia_,
            "silhouette_score": sil,
        })

    metrics_df = pd.DataFrame(records)
    optimal_k = int(metrics_df.loc[metrics_df["silhouette_score"].idxmax(), "k"])
    return optimal_k, metrics_df
