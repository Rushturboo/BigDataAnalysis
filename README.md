# BigDataAnalysis

> BDT4864A 大数据分析实践 — 基于机器学习的大数据分析与可视化

## 📋 项目信息

| 项目     | 内容                    |
| -------- | ----------------------- |
| 课程     | BDT4864A 大数据分析实践 |
| 指导教师 | 陈亮、汪飞              |
| 周期     | 2026.07.06 - 2026.07.20 |
| 选题     | [待确定]                |

## 👥 组员

| 姓名   | 角色 | 分工 |
| ------ | ---- | ---- |
| 吴彦弘 | 组长 |      |
| 陆嘉怡 | 组员 |      |
| 全芷莹 | 组员 |      |
| 黄家强 | 组员 |      |
| 陈泓旭 | 组员 |      |

## 📁 目录结构

```
BigDataAnalysis/
├── data/                  # 数据文件
│   ├── raw/              # 原始数据（不上传Git）
│   └── processed/        # 处理后数据（不上传Git）
├── notebooks/            # Jupyter Notebooks
├── src/                  # 源代码
│   ├── data_preprocessing/  # 数据预处理
│   ├── analysis/            # 分析与建模
│   └── visualization/       # 可视化
├── reports/              # 报告文档
├── docs/                 # 项目文档
├── models/               # 训练好的模型
├── requirements.txt      # Python 依赖
├── .gitignore
└── README.md
```

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/[你的用户名]/BigDataAnalysis.git

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
