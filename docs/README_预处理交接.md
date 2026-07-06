# 预处理数据使用说明

> **预处理负责人**: 黄家强  
> **完成时间**: 2026-07-06  
> **运行环境**: `conda activate traffic_env`  
> **预处理脚本**: `src/data_preprocessing/preprocess.py`

---

## 一、如何生成预处理数据

```bash
# 1. 把原始数据放到 data/raw/MSBB.366.samples_SE/
# 2. 运行预处理
conda activate traffic_env
python src/data_preprocessing/preprocess.py
```

输出在 `data/processed/`（该目录不上传 Git，每人本地运行生成）。

---

## 二、输出文件一览

| 文件 | 说明 |
|------|------|
| `BM_36_logcpm_matrix.csv` | **推荐优先使用**。BM_36（海马旁回）表达矩阵 |
| `BM_36_sample_metadata.csv` | BM_36 每个样本的临床信息 |
| `recommended_BM36_*.csv` | BM_36 快捷副本 |
| `BM_10/22/44_logcpm_matrix.csv` | 其他脑区表达矩阵 |
| `all_samples_metadata.csv` | 全部 938 个样本元数据合并 |
| `preprocessing_log.txt` | 预处理过程日志 |

---

## 三、预处理统计

| 脑区 | 样本数 | 基因数（过滤后） | 映射率 |
|------|--------|-----------------|--------|
| BM_10（额极） | 261 | 52,539 | 100% |
| BM_22（颞上回） | 240 | 52,870 | 100% |
| **BM_36（海马旁回）** | **215** | **48,553** | **100%** |
| BM_44（额下回） | 222 | 46,473 | 100% |

- 过滤：删除在少于 10 个样本中有表达的基因
- 标准化：CPM + log2(x+1)
- 合计 938 样本，299 个独立个体

---

## 四、下游读入示例

```python
import pandas as pd

expr = pd.read_csv("data/processed/BM_36_logcpm_matrix.csv", index_col=0)
meta = pd.read_csv("data/processed/BM_36_sample_metadata.csv", index_col=0)
```

### 特征选择 → 降维 → 聚类

```python
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap

top_genes = expr.var(axis=1).nlargest(3000).index
X = expr.loc[top_genes].T
X_pca = PCA(n_components=50, random_state=42).fit_transform(X)
embedding = umap.UMAP(n_components=2, random_state=42).fit_transform(X_pca)
clusters = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_pca)
```

### 散点图 / 火山图 / 热图

详见原文档代码，路径把 `output/preprocessed/` 改为 `data/processed/` 即可。

---



*预处理数据文件较大，通过飞书/网盘共享，Git 只存代码。*
