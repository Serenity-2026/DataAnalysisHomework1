"""任务二：无泄漏的数据清洗、编码、划分与缩放。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_PATH,
    PREPROCESSING_OUTPUT_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from .quality import load_raw_data


NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
TARGET_MAPPING = {"No": 0, "Yes": 1}


@dataclass
class PreparedData:
    """向后续算法提供同一份划分及缩放/未缩放数据。"""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_scaled: pd.DataFrame
    X_test_scaled: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    scaler: StandardScaler
    numeric_imputer: SimpleImputer
    categorical_imputer: SimpleImputer
    metadata: dict[str, Any]


def normalize_string_values(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """去除字符串首尾空格，并将空字符串规范为缺失值。"""

    normalized = data.copy(deep=True)
    normalized_counts: dict[str, int] = {}
    object_columns = normalized.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        series = normalized[column].astype("string")
        stripped = series.str.strip()
        changed = series.notna() & series.ne(stripped)
        blank = series.notna() & stripped.eq("")
        normalized_counts[column] = int((changed | blank).sum())
        normalized[column] = stripped.mask(stripped.eq(""), pd.NA)
    return normalized, normalized_counts


def _clean_supervised_rows(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """删除全空、缺少标签和完全重复的记录，并保留计数。"""

    cleaned = data.copy(deep=True)
    all_empty_mask = cleaned.isna().all(axis=1)
    all_empty_rows_removed = int(all_empty_mask.sum())
    cleaned = cleaned.loc[~all_empty_mask].copy()

    missing_target_mask = cleaned["Churn"].isna()
    missing_target_rows_removed = int(missing_target_mask.sum())
    severely_incomplete_target_rows = int(
        (
            missing_target_mask
            & cleaned.isna().sum(axis=1).ge(len(cleaned.columns) - 2)
        ).sum()
    )
    cleaned = cleaned.loc[~missing_target_mask].copy()

    duplicate_mask = cleaned.duplicated(keep="first")
    duplicate_rows_removed = int(duplicate_mask.sum())
    cleaned = cleaned.loc[~duplicate_mask].copy()

    customer_id_missing = int(cleaned["customerID"].isna().sum())
    customer_id_duplicates = int(cleaned["customerID"].duplicated().sum())
    if customer_id_missing or customer_id_duplicates:
        raise ValueError(
            "customerID 必须完整且唯一；"
            f"缺失={customer_id_missing}，重复={customer_id_duplicates}。"
            "请人工核对，预处理不会擅自删除这些记录。"
        )

    return cleaned, {
        "all_empty_rows_removed": all_empty_rows_removed,
        "missing_target_rows_removed": missing_target_rows_removed,
        "severely_incomplete_missing_target_rows": severely_incomplete_target_rows,
        "fully_duplicated_rows_removed": duplicate_rows_removed,
    }


def _convert_numeric_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """转换数值字段，并按业务规则处理 tenure=0 的累计费用缺失。"""

    converted = data.copy(deep=True)
    total_before = converted["TotalCharges"].astype("string")
    nonblank_total = total_before.notna() & total_before.str.strip().ne("")

    for column in NUMERIC_FEATURES:
        converted[column] = pd.to_numeric(converted[column], errors="coerce")

    total_conversion_failures = int(
        (nonblank_total & converted["TotalCharges"].isna()).sum()
    )
    tenure_zero_total_missing = converted["TotalCharges"].isna() & converted[
        "tenure"
    ].eq(0)
    tenure_zero_filled = int(tenure_zero_total_missing.sum())
    converted.loc[tenure_zero_total_missing, "TotalCharges"] = 0.0

    return converted, {
        "total_charges_nonblank_conversion_failures": total_conversion_failures,
        "total_charges_tenure_zero_filled_with_zero": tenure_zero_filled,
        "total_charges_remaining_missing_before_split": int(
            converted["TotalCharges"].isna().sum()
        ),
    }


def _validate_and_map_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """验证 Churn 只有 Yes/No，并映射为 0/1。"""

    labels = data["Churn"].astype("string").str.strip()
    unknown = sorted(set(labels.dropna().unique()) - set(TARGET_MAPPING))
    if unknown:
        raise ValueError(f"Churn 存在未识别标签：{unknown}")
    mapped = labels.map(TARGET_MAPPING)
    if mapped.isna().any():
        raise ValueError("Churn 映射后仍含缺失值，请检查原始标签。")
    return data.drop(columns=["Churn"]), mapped.astype("int64").rename("Churn")


def _impute_from_training(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    SimpleImputer,
    SimpleImputer,
    dict[str, list[str]],
    dict[str, int],
]:
    """只用训练集拟合缺失值规则，并为独热编码固定训练类别。"""

    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")

    train_numeric = pd.DataFrame(
        numeric_imputer.fit_transform(X_train[numeric_columns]),
        columns=numeric_columns,
        index=X_train.index,
    )
    test_numeric = pd.DataFrame(
        numeric_imputer.transform(X_test[numeric_columns]),
        columns=numeric_columns,
        index=X_test.index,
    )

    train_categorical_input = X_train[categorical_columns].astype(object)
    test_categorical_input = X_test[categorical_columns].astype(object)
    train_categorical_input = train_categorical_input.where(
        train_categorical_input.notna(), np.nan
    )
    test_categorical_input = test_categorical_input.where(
        test_categorical_input.notna(), np.nan
    )

    train_categorical = pd.DataFrame(
        categorical_imputer.fit_transform(train_categorical_input),
        columns=categorical_columns,
        index=X_train.index,
    ).astype("string")
    test_categorical = pd.DataFrame(
        categorical_imputer.transform(test_categorical_input),
        columns=categorical_columns,
        index=X_test.index,
    ).astype("string")

    category_levels: dict[str, list[str]] = {}
    unknown_test_categories: dict[str, int] = {}
    for column in categorical_columns:
        levels = sorted(train_categorical[column].dropna().unique().tolist())
        if not levels:
            raise ValueError(f"训练集类别字段 {column} 没有可用取值。")
        category_levels[column] = [str(level) for level in levels]
        unknown_test_categories[column] = int(
            (~test_categorical[column].isin(levels)).sum()
        )
        train_categorical[column] = pd.Categorical(
            train_categorical[column], categories=levels
        )
        test_categorical[column] = pd.Categorical(
            test_categorical[column], categories=levels
        )

    train_dummies = pd.get_dummies(
        train_categorical,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    )
    test_dummies = pd.get_dummies(
        test_categorical,
        columns=categorical_columns,
        drop_first=True,
        dtype=float,
    ).reindex(columns=train_dummies.columns, fill_value=0.0)

    encoded_train = pd.concat([train_numeric, train_dummies], axis=1).astype(float)
    encoded_test = pd.concat([test_numeric, test_dummies], axis=1).astype(float)
    encoded_test = encoded_test.reindex(columns=encoded_train.columns, fill_value=0.0)

    return (
        encoded_train,
        encoded_test,
        numeric_imputer,
        categorical_imputer,
        category_levels,
        unknown_test_categories,
    )


def _label_ratio(labels: pd.Series) -> dict[str, float]:
    return {
        str(int(label)): float(ratio)
        for label, ratio in labels.value_counts(normalize=True).sort_index().items()
    }


def prepare_data_from_frame(
    raw_data: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """从 DataFrame 生成统一数据接口；所有拟合步骤只使用训练集。"""

    if not 0 < test_size < 1:
        raise ValueError(f"test_size 必须在 (0, 1) 内，实际为 {test_size}")

    normalized, normalization_counts = normalize_string_values(raw_data)
    cleaned_rows, row_report = _clean_supervised_rows(normalized)
    converted, numeric_report = _convert_numeric_columns(cleaned_rows)
    features_with_id, target = _validate_and_map_target(converted)

    features = features_with_id.drop(columns=["customerID"])
    categorical_columns = [
        column for column in features.columns if column not in NUMERIC_FEATURES
    ]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    (
        X_train,
        X_test,
        numeric_imputer,
        categorical_imputer,
        category_levels,
        unknown_test_categories,
    ) = _impute_from_training(
        X_train_raw,
        X_test_raw,
        NUMERIC_FEATURES,
        categorical_columns,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    arrays = [X_train, X_test, X_train_scaled, X_test_scaled]
    if any(not np.isfinite(frame.to_numpy(dtype=float)).all() for frame in arrays):
        raise ValueError("预处理结果包含 NaN 或无穷值。")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise AssertionError("训练集和测试集编码列不一致。")

    scaler_samples = int(np.asarray(scaler.n_samples_seen_).max())
    metadata: dict[str, Any] = {
        "raw_shape": {
            "rows": int(raw_data.shape[0]),
            "columns": int(raw_data.shape[1]),
        },
        "clean_shape_before_split": {
            "rows": int(len(features)),
            "feature_columns_before_encoding": int(features.shape[1]),
        },
        "normalization_counts": normalization_counts,
        "row_cleaning": row_report,
        "numeric_cleaning": numeric_report,
        "split": {
            "test_size": float(test_size),
            "random_state": int(random_state),
            "stratified": True,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_label_ratio": _label_ratio(y_train),
            "test_label_ratio": _label_ratio(y_test),
        },
        "encoding": {
            "method": "pandas.get_dummies",
            "drop_first": True,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": categorical_columns,
            "category_levels_fitted_on_training": category_levels,
            "unknown_test_categories": unknown_test_categories,
            "feature_count": int(X_train.shape[1]),
        },
        "imputation": {
            "numeric_strategy": "training median",
            "categorical_strategy": "training most_frequent",
            "numeric_statistics": {
                column: float(value)
                for column, value in zip(NUMERIC_FEATURES, numeric_imputer.statistics_)
            },
            "categorical_statistics": {
                column: str(value)
                for column, value in zip(
                    categorical_columns, categorical_imputer.statistics_
                )
            },
        },
        "scaling": {
            "method": "StandardScaler",
            "fit_on_training_only": True,
            "n_samples_seen": scaler_samples,
        },
    }

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_names=X_train.columns.tolist(),
        scaler=scaler,
        numeric_imputer=numeric_imputer,
        categorical_imputer=categorical_imputer,
        metadata=metadata,
    )


def prepare_data(
    path: str | Path = DATA_PATH,
    *,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """读取课程 CSV 并返回统一的 PreparedData。"""

    return prepare_data_from_frame(
        load_raw_data(path), test_size=test_size, random_state=random_state
    )


def _write_generated(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"输出已存在：{path}；如需重新生成请传入 --force。")
    path.write_text(text, encoding="utf-8")


def save_preprocessing_outputs(
    prepared: PreparedData,
    output_dir: str | Path = PREPROCESSING_OUTPUT_DIR,
    *,
    force: bool = False,
) -> list[Path]:
    """保存不含客户记录的元数据、特征清单和真实运行日志。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    summary_path = destination / "preprocessing_summary.json"
    feature_path = destination / "feature_names.txt"
    log_path = destination / "preprocessing_run.log"

    summary_text = json.dumps(prepared.metadata, ensure_ascii=False, indent=2) + "\n"
    feature_text = "\n".join(prepared.feature_names) + "\n"
    log_lines = [
        "A 数据工程师任务二真实运行记录",
        f"训练样本数: {len(prepared.X_train)}",
        f"测试样本数: {len(prepared.X_test)}",
        f"编码后特征数: {len(prepared.feature_names)}",
        f"训练标签比例: {prepared.metadata['split']['train_label_ratio']}",
        f"测试标签比例: {prepared.metadata['split']['test_label_ratio']}",
        f"scaler 拟合样本数: {prepared.metadata['scaling']['n_samples_seen']}",
        "训练/测试编码列一致: True",
        "四套模型输入均无 NaN/无穷值: True",
    ]
    log_text = "\n".join(log_lines) + "\n"

    _write_generated(summary_path, summary_text, force=force)
    _write_generated(feature_path, feature_text, force=force)
    _write_generated(log_path, log_text, force=force)
    return [summary_path, feature_path, log_path]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行客户流失数据预处理")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="原始 CSV 路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREPROCESSING_OUTPUT_DIR,
        help="元数据输出目录",
    )
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--force", action="store_true", help="覆盖由本脚本生成的同名输出"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    prepared = prepare_data(
        args.data, test_size=args.test_size, random_state=args.random_state
    )
    paths = save_preprocessing_outputs(prepared, args.output_dir, force=args.force)
    print("预处理完成。")
    for path in paths:
        print(f"已生成: {path}")
    print(
        f"训练={prepared.X_train.shape}，测试={prepared.X_test.shape}，"
        f"scaler.n_samples_seen_={prepared.metadata['scaling']['n_samples_seen']}"
    )


if __name__ == "__main__":
    main()
