"""
MSBB 阿尔茨海默病 RNA-seq 数据预处理
负责人: 黄家强
环境: conda activate traffic_env

数据路径（团队仓库标准）:
  data/raw/MSBB.366.samples_SE/        ← 原始数据放这里
  data/processed/                      ← 预处理输出（不上传 Git，本地生成）
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── 路径配置（适配 BigDataAnalysis 仓库结构）──────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "MSBB.366.samples_SE"
EXPR_DIR = DATA_ROOT / "Processed" / "Quantitated.Expression"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

BRAIN_AREAS = {
    "BM_10": "AMP-AD_MSBB_MSSM_BM_10.raw_counts.tsv",
    "BM_22": "AMP-AD_MSBB_MSSM_BM_22.raw_counts.tsv",
    "BM_36": "AMP-AD_MSBB_MSSM_BM_36.raw_counts.tsv",
    "BM_44": "AMP-AD_MSBB_MSSM_BM_44.raw_counts.tsv",
}

MIN_SAMPLES = 10
CLINICAL_COLS = [
    "individualID",
    "sex",
    "race",
    "ageDeath",
    "apoeGenotype",
    "pmi",
    "CERAD",
    "Braak",
    "CDR",
]


def build_specimen_lookup(biospec: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for _, row in biospec.iterrows():
        sid = row["specimenID"]
        iid = row["individualID"]
        if pd.isna(sid):
            continue
        sid = str(sid)
        lookup[sid] = iid

        bm_match = re.search(r"(BM_\d+_\d+)", sid)
        if bm_match:
            lookup[bm_match.group(1)] = iid

        rna_match = re.search(r"hB_RNA_(\d+)", sid)
        if rna_match:
            lookup[f"hB_RNA_{rna_match.group(1)}"] = iid

    return lookup


def map_sample_to_individual(
    sample_id: str, lookup: dict[str, str]
) -> tuple[str | None, str | None]:
    bm_match = re.search(r"(BM_\d+_\d+)", sample_id)
    if bm_match and bm_match.group(1) in lookup:
        key = bm_match.group(1)
        return lookup[key], key

    rna_match = re.search(r"hB_RNA_(\d+)", sample_id)
    if rna_match:
        key = f"hB_RNA_{rna_match.group(1)}"
        if key in lookup:
            return lookup[key], key

    return None, None


def filter_low_expression(expr: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    keep = (expr > 0).sum(axis=1) >= min_samples
    return expr.loc[keep]


def cpm_log2(expr: pd.DataFrame) -> pd.DataFrame:
    lib_sizes = expr.sum(axis=0)
    cpm = expr.div(lib_sizes, axis=1) * 1e6
    return np.log2(cpm + 1)


def build_sample_metadata(
    sample_ids: list[str],
    brain_area: str,
    lookup: dict[str, str],
    clinical: pd.DataFrame,
) -> pd.DataFrame:
    clinical = clinical.set_index("individualID")
    records = []

    for sample_id in sample_ids:
        individual_id, specimen_key = map_sample_to_individual(sample_id, lookup)
        row = {
            "sample_id": sample_id,
            "brain_area": brain_area,
            "specimen_key": specimen_key,
            "individualID": individual_id,
            "has_clinical": individual_id in clinical.index if individual_id else False,
        }

        if individual_id and individual_id in clinical.index:
            for col in CLINICAL_COLS:
                if col != "individualID":
                    row[col] = clinical.loc[individual_id, col]
        else:
            for col in CLINICAL_COLS:
                if col != "individualID":
                    row[col] = np.nan

        records.append(row)

    return pd.DataFrame(records).set_index("sample_id")


def process_brain_area(
    brain_area: str,
    filename: str,
    lookup: dict[str, str],
    clinical: pd.DataFrame,
    min_samples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = EXPR_DIR / filename
    print(f"\n[{brain_area}] 读取 {path.name} ...")
    expr = pd.read_csv(path, sep="\t", index_col=0)

    n_genes_raw, n_samples = expr.shape
    expr_filtered = filter_low_expression(expr, min_samples)
    expr_norm = cpm_log2(expr_filtered)

    metadata = build_sample_metadata(
        list(expr.columns), brain_area, lookup, clinical
    )

    mapped = metadata["individualID"].notna().sum()
    with_clinical = metadata["has_clinical"].sum()

    stats = {
        "brain_area": brain_area,
        "n_samples": int(n_samples),
        "n_genes_raw": int(n_genes_raw),
        "n_genes_filtered": int(expr_filtered.shape[0]),
        "genes_removed": int(n_genes_raw - expr_filtered.shape[0]),
        "samples_mapped": int(mapped),
        "samples_with_clinical": int(with_clinical),
        "mapping_rate": round(mapped / n_samples * 100, 1),
    }

    print(
        f"  样本 {n_samples} | 基因 {n_genes_raw}→{expr_filtered.shape[0]} "
        f"| 映射 {mapped}/{n_samples} | 有临床信息 {with_clinical}"
    )

    return expr_norm, metadata, stats


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MSBB 数据预处理")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    biospec = pd.read_csv(DATA_ROOT / "MSBB_biospecimen_metadata.csv")
    clinical = pd.read_csv(DATA_ROOT / "msbb_individual_metadata.csv")
    lookup = build_specimen_lookup(biospec)

    all_metadata: list[pd.DataFrame] = []
    all_stats: list[dict] = []

    for brain_area, filename in BRAIN_AREAS.items():
        expr_norm, metadata, stats = process_brain_area(
            brain_area, filename, lookup, clinical, MIN_SAMPLES
        )

        matrix_path = OUTPUT_DIR / f"{brain_area}_logcpm_matrix.csv"
        meta_path = OUTPUT_DIR / f"{brain_area}_sample_metadata.csv"

        print(f"  保存 {matrix_path.name} ...")
        expr_norm.to_csv(matrix_path)
        metadata.to_csv(meta_path)

        all_metadata.append(metadata)
        all_stats.append(stats)

    combined_meta = pd.concat(all_metadata)
    combined_meta.to_csv(OUTPUT_DIR / "all_samples_metadata.csv")

    import shutil

    shutil.copy(
        OUTPUT_DIR / "BM_36_logcpm_matrix.csv",
        OUTPUT_DIR / "recommended_BM36_logcpm_matrix.csv",
    )
    shutil.copy(
        OUTPUT_DIR / "BM_36_sample_metadata.csv",
        OUTPUT_DIR / "recommended_BM36_sample_metadata.csv",
    )

    summary = {
        "preprocessing_date": datetime.now().isoformat(timespec="seconds"),
        "responsible": "黄家强",
        "min_samples_threshold": MIN_SAMPLES,
        "normalization": "CPM + log2(x+1)",
        "brain_areas": all_stats,
        "total_samples": int(combined_meta.shape[0]),
        "unique_individuals": int(combined_meta["individualID"].nunique()),
    }

    with open(OUTPUT_DIR / "preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    log_lines = [
        "MSBB 数据预处理日志",
        f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "负责人: 黄家强",
        f"过滤阈值: 基因至少在 {MIN_SAMPLES} 个样本中有表达",
        "标准化: CPM + log2(x+1)",
        "",
        "各脑区统计:",
    ]
    for s in all_stats:
        log_lines.append(
            f"  {s['brain_area']}: "
            f"{s['n_samples']} 样本, "
            f"{s['n_genes_raw']}→{s['n_genes_filtered']} 基因, "
            f"映射率 {s['mapping_rate']}%"
        )
    log_lines.extend(
        [
            "",
            f"合计样本数: {combined_meta.shape[0]}",
            f"独立个体数: {combined_meta['individualID'].nunique()}",
        ]
    )

    (OUTPUT_DIR / "preprocessing_log.txt").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )

    readme_src = PROJECT_ROOT / "docs" / "README_预处理交接.md"
    if readme_src.exists():
        shutil.copy(readme_src, OUTPUT_DIR / "README_给下游同学.md")

    print("\n" + "=" * 60)
    print("预处理完成!")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
