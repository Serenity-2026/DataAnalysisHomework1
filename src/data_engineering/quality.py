"""读取并审计原始客户流失数据。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPECTED_COLUMNS


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """读取 CSV，并验证任务书规定的字段集合和顺序。"""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")

    data = pd.read_csv(csv_path)
    actual_columns = data.columns.tolist()
    if actual_columns != EXPECTED_COLUMNS:
        missing = [column for column in EXPECTED_COLUMNS if column not in actual_columns]
        extra = [column for column in actual_columns if column not in EXPECTED_COLUMNS]
        raise ValueError(
            "CSV 字段与任务书不一致。"
            f"缺少字段={missing}，额外字段={extra}，实际顺序={actual_columns}"
        )
    return data


def _object_columns(data: pd.DataFrame) -> list[str]:
    return data.select_dtypes(include=["object", "string"]).columns.tolist()


def build_quality_summary(data: pd.DataFrame) -> dict[str, Any]:
    """返回 JSON 可序列化的数据质量计数，不修改输入数据。"""

    object_columns = _object_columns(data)
    missing_count = data.isna().sum()
    missing_ratio = data.isna().mean()

    blank_counts: dict[str, int] = {}
    surrounding_whitespace_counts: dict[str, int] = {}
    for column in object_columns:
        series = data[column].astype("string")
        non_missing = series.notna()
        stripped = series.str.strip()
        blank_counts[column] = int((non_missing & stripped.eq("")).sum())
        surrounding_whitespace_counts[column] = int(
            (non_missing & series.ne(stripped)).sum()
        )

    total_charges_text = data["TotalCharges"].astype("string")
    total_charges_stripped = total_charges_text.str.strip()
    total_charges_blank = total_charges_text.notna() & total_charges_stripped.eq("")
    total_charges_numeric = pd.to_numeric(
        total_charges_stripped.mask(total_charges_blank), errors="coerce"
    )
    total_charges_conversion_failures = (
        total_charges_text.notna()
        & total_charges_stripped.ne("")
        & total_charges_numeric.isna()
    )

    customer_ids = data["customerID"].astype("string")
    customer_id_blank = customer_ids.notna() & customer_ids.str.strip().eq("")

    # 任务数据最后一行只保留 customerID 和残缺的 gender 文本。普通 all(axis=1)
    # 不会把它识别为全空，因此同时记录“缺少标签”的监督学习无效行。
    all_empty_mask = data.isna().all(axis=1)
    missing_target_mask = data["Churn"].isna()
    severely_incomplete_mask = data.isna().sum(axis=1).ge(len(data.columns) - 2)

    churn_counts = data["Churn"].value_counts(dropna=False)
    churn_count_dict = {
        ("<NA>" if pd.isna(label) else str(label)): int(count)
        for label, count in churn_counts.items()
    }
    valid_churn = data["Churn"].dropna().astype("string").str.strip()
    churn_ratios = valid_churn.value_counts(normalize=True)

    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "columns": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing": {
            column: {
                "count": int(missing_count[column]),
                "ratio": float(missing_ratio[column]),
            }
            for column in data.columns
        },
        "blank_strings_after_strip": blank_counts,
        "values_with_surrounding_whitespace": surrounding_whitespace_counts,
        "all_empty_rows": int(all_empty_mask.sum()),
        "missing_target_rows": int(missing_target_mask.sum()),
        "severely_incomplete_rows": int(severely_incomplete_mask.sum()),
        "fully_duplicated_rows": int(data.duplicated().sum()),
        "customer_id": {
            "missing": int(data["customerID"].isna().sum()),
            "blank": int(customer_id_blank.sum()),
            "duplicated_non_missing": int(
                data.loc[data["customerID"].notna(), "customerID"].duplicated().sum()
            ),
        },
        "total_charges": {
            "original_dtype": str(data["TotalCharges"].dtype),
            "blank_strings": int(total_charges_blank.sum()),
            "nonblank_conversion_failures": int(
                total_charges_conversion_failures.sum()
            ),
            "numeric_missing_after_conversion": int(total_charges_numeric.isna().sum()),
            "numeric_missing_with_tenure_zero": int(
                (total_charges_numeric.isna() & data["tenure"].eq(0)).sum()
            ),
            "numeric_missing_with_tenure_nonzero": int(
                (total_charges_numeric.isna() & ~data["tenure"].eq(0)).sum()
            ),
        },
        "churn": {
            "counts_including_missing": churn_count_dict,
            "valid_ratios": {
                str(label): float(ratio) for label, ratio in churn_ratios.items()
            },
            "unexpected_non_missing_labels": sorted(
                set(valid_churn.dropna().unique()) - {"Yes", "No"}
            ),
        },
    }
