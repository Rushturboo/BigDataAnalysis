"""项目路径与常量配置。"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# 数据路径
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MSBB_DIR = DATA_RAW / "MSBB.366.samples_SE"
EXPR_DIR = MSBB_DIR / "Processed" / "Quantitated.Expression"

BRAIN_REGIONS = {
    "BM_10": EXPR_DIR / "AMP-AD_MSBB_MSSM_BM_10.raw_counts.tsv",
    "BM_22": EXPR_DIR / "AMP-AD_MSBB_MSSM_BM_22.raw_counts.tsv",
    "BM_36": EXPR_DIR / "AMP-AD_MSBB_MSSM_BM_36.raw_counts.tsv",
    "BM_44": EXPR_DIR / "AMP-AD_MSBB_MSSM_BM_44.raw_counts.tsv",
}

CLINICAL_FILE = MSBB_DIR / "msbb_individual_metadata.csv"
BIOSPECIMEN_FILE = MSBB_DIR / "MSBB_biospecimen_metadata.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

RANDOM_SEED = 42
