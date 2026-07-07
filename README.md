# BigDataAnalysis

> BDT4864A 大数据分析实践 — AD 分亚型：基于 MSBB 转录组数据的无监督聚类与可视化

## 📋 项目信息

| 项目     | 内容                    |
| -------- | ----------------------- |
| 课程     | BDT4864A 大数据分析实践 |
| 指导教师 | 陈亮、汪飞              |
| 周期     | 2026.07.06 - 2026.07.20 |
| 选题     | AD（阿尔茨海默病）分亚型 |
| 数据来源 | MSBB (Mount Sinai Brain Bank) / AMP-AD 联盟 |


## 📁 目录结构

```
BigDataAnalysis/
├── data/                     # 数据文件
│   ├── raw/                 # 原始数据（不上传Git）
│   └── processed/           # 处理后数据（不上传Git）
├── notebooks/               # Jupyter Notebooks
│   ├── 01_eda.ipynb         # 数据探索
│   ├── 02_preprocessing.ipynb  # 预处理验证
│   ├── 03_clustering_3d.ipynb  # 图1：3D聚类散点图
│   ├── 04_volcano.ipynb     # 图2：火山图
│   └── 05_network.ipynb     # 图3：共表达网络图
├── src/                     # 源代码
│   ├── data_preprocessing/  # 数据预处理
│   ├── analysis/            # 分析与建模
│   │   ├── clustering.py    # 图1：PCA→UMAP→KMeans
│   │   ├── differential.py  # 图2：差异表达分析
│   │   └── network.py       # 图3：共表达网络
│   ├── visualization/       # 可视化
│   │   ├── scatter3d.py     # 图1：3D聚类散点图
│   │   ├── volcano.py       # 图2：火山图
│   │   └── network_viz.py   # 图3：3D网络图
│   └── utils/
│       └── config.py        # 公共路径/常量
├── outputs/figures/         # 图表输出
├── docs/                    # 项目文档
├── requirements.txt         # Python 依赖
├── .gitignore
└── README.md
```

## 🔬 三张核心图

| 图 | 内容 | 分析方法 | 可视化 |
|---|---|---|---|
| 图 1 | 3D 聚类散点图 | PCA → UMAP → K-means | Plotly 3D 交互 |
| 图 2 | 火山图 | 亚型间 t-test 差异表达 | Plotly 标注 |
| 图 3 | 基因共表达网络 | 相关矩阵 → WGCNA 模块 | Plotly 3D 网络 |

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/Rushturboo/BigDataAnalysis.git

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

## 📅 关键时间节点

- **7/06** — 任务布置，分组完成
- **7/13** — 中期检查（阶段性验收，提交报告草稿）
- **7/20** — 终提交（报告 + PPT + 源码数据 + 会议记录）

## 🔒 安全说明

- 严禁将 API Key、密码等敏感信息提交到 Git
- 所有敏感配置请放在 `.env` 文件中（已加入 .gitignore）
- 大文件数据不要直接 push，使用 .gitignore 排除
