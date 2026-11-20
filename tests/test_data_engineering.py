"""任务一和任务二的数据质量与无泄漏预处理测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_engineering.config import EXPECTED_COLUMNS
from src.data_engineering.preprocessing import (
    normalize_string_values,
    prepare_data_from_frame,
)
from src.data_engineering.quality import build_quality_summary, load_raw_data


def make_sample_data(*, include_invalid_rows: bool = True) -> pd.DataFrame:
    """构造仅用于边界测试的小样例，不代表课程数据或实验结果。"""

    rows: list[dict[str, object]] = []
    for index in range(20):
        tenure = 0 if index == 0 else index + 1
        total_charges: object = str(round((index + 1) * 25.5, 2))
        if index == 0:
            total_charges = "   "
        elif index == 1:
            total_charges = "not-a-number"
        rows.append(
            {
                "customerID": f"C{index:04d}",
                "gender": " Female " if index % 2 else "Male",
                "SeniorCitizen": index % 2,
                "Partner": "Yes" if index % 3 else "No",
                "Dependents": "No" if index % 4 else "Yes",
                "tenure": tenure,
                "PhoneService": "Yes",
                "MultipleLines": "No" if index % 2 else "Yes",
                "InternetService": "DSL" if index % 2 else "Fiber optic",
                "OnlineSecurity": "No" if index % 2 else "Yes",
                "OnlineBackup": "Yes" if index % 2 else "No",
                "DeviceProtection": "No",
                "TechSupport": "No" if index % 2 else "Yes",
                "StreamingTV": "Yes" if index % 3 else "No",
                "StreamingMovies": "No" if index % 3 else "Yes",
                "Contract": "Month-to-month" if index % 2 else "One year",
                "PaperlessBilling": "Yes" if index % 2 else "No",
                "PaymentMethod": "Electronic check" if index % 2 else "Mailed check",
                "MonthlyCharges": 50.0 + index,
                "TotalCharges": total_charges,
                "Churn": "Yes" if index % 2 else "No",
            }
        )

    if include_invalid_rows:
        rows.append({column: np.nan for column in EXPECTED_COLUMNS})
        truncated = {column: np.nan for column in EXPECTED_COLUMNS}
        truncated["customerID"] = "TRUNCATED"
        truncated["gender"] = "Fem"
        rows.append(truncated)

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


class QualityTests(unittest.TestCase):
    def test_load_raw_data_validates_columns(self) -> None:
        data = make_sample_data(include_invalid_rows=False)
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.csv"
            data.to_csv(valid_path, index=False)
            loaded = load_raw_data(valid_path)
            self.assertEqual(loaded.columns.tolist(), EXPECTED_COLUMNS)

            invalid_path = Path(directory) / "invalid.csv"
            data.drop(columns=["Churn"]).to_csv(invalid_path, index=False)
            with self.assertRaisesRegex(ValueError, "字段与任务书不一致"):
                load_raw_data(invalid_path)

    def test_quality_summary_finds_hidden_blanks_and_invalid_rows(self) -> None:
        summary = build_quality_summary(make_sample_data())
        self.assertEqual(summary["blank_strings_after_strip"]["TotalCharges"], 1)
        self.assertEqual(summary["all_empty_rows"], 1)
        self.assertEqual(summary["missing_target_rows"], 2)
        self.assertEqual(summary["severely_incomplete_rows"], 2)
        self.assertEqual(summary["fully_duplicated_rows"], 0)
        self.assertEqual(summary["total_charges"]["nonblank_conversion_failures"], 1)


class PreprocessingTests(unittest.TestCase):
    def test_normalize_strings_strips_and_converts_blank(self) -> None:
        data = make_sample_data(include_invalid_rows=False)
        normalized, counts = normalize_string_values(data)
        self.assertEqual(normalized.loc[1, "gender"], "Female")
        self.assertTrue(pd.isna(normalized.loc[0, "TotalCharges"]))
        self.assertEqual(counts["gender"], 10)
        self.assertEqual(counts["TotalCharges"], 1)

    def test_prepare_removes_invalid_rows_and_maps_target(self) -> None:
        prepared = prepare_data_from_frame(make_sample_data(), test_size=0.2)
        row_report = prepared.metadata["row_cleaning"]
        self.assertEqual(row_report["all_empty_rows_removed"], 1)
        self.assertEqual(row_report["missing_target_rows_removed"], 1)
        self.assertEqual(
            row_report["severely_incomplete_missing_target_rows"], 1
        )
        self.assertEqual(set(prepared.y_train.unique()), {0, 1})
        self.assertEqual(set(prepared.y_test.unique()), {0, 1})

    def test_total_charges_rules_and_training_median(self) -> None:
        raw = make_sample_data()
        prepared = prepare_data_from_frame(raw, test_size=0.2, random_state=42)
        numeric_report = prepared.metadata["numeric_cleaning"]
        self.assertEqual(numeric_report["total_charges_tenure_zero_filled_with_zero"], 1)
        self.assertEqual(numeric_report["total_charges_nonblank_conversion_failures"], 1)

        train_indices = prepared.X_train.index
        training_total = pd.to_numeric(
            raw.loc[train_indices, "TotalCharges"].astype("string").str.strip(),
            errors="coerce",
        )
        tenure_zero = raw.loc[train_indices, "tenure"].eq(0)
        training_total.loc[training_total.isna() & tenure_zero] = 0.0
        expected_training_median = float(training_total.median())
        actual_statistic = prepared.metadata["imputation"]["numeric_statistics"][
            "TotalCharges"
        ]
        self.assertAlmostEqual(actual_statistic, expected_training_median)

    def test_unknown_target_raises_clear_error(self) -> None:
        data = make_sample_data(include_invalid_rows=False)
        data.loc[0, "Churn"] = "Maybe"
        with self.assertRaisesRegex(ValueError, "未识别标签"):
            prepare_data_from_frame(data)

    def test_duplicate_customer_id_requires_manual_review(self) -> None:
        data = make_sample_data(include_invalid_rows=False)
        data.loc[1, "customerID"] = data.loc[0, "customerID"]
        with self.assertRaisesRegex(ValueError, "customerID 必须完整且唯一"):
            prepare_data_from_frame(data)

    def test_interface_columns_scaling_and_finite_values(self) -> None:
        raw = make_sample_data()
        raw_copy = raw.copy(deep=True)
        prepared = prepare_data_from_frame(raw, test_size=0.2, random_state=42)

        pd.testing.assert_frame_equal(raw, raw_copy)
        self.assertNotIn("customerID", prepared.X_train.columns)
        self.assertEqual(prepared.X_train.columns.tolist(), prepared.feature_names)
        self.assertEqual(
            prepared.X_train.columns.tolist(), prepared.X_test.columns.tolist()
        )
        self.assertEqual(
            prepared.X_train_scaled.columns.tolist(), prepared.feature_names
        )
        self.assertEqual(
            prepared.X_test_scaled.columns.tolist(), prepared.feature_names
        )

        for frame in [
            prepared.X_train,
            prepared.X_test,
            prepared.X_train_scaled,
            prepared.X_test_scaled,
        ]:
            self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())
            self.assertFalse(frame.isna().any().any())

        self.assertEqual(
            prepared.metadata["scaling"]["n_samples_seen"],
            len(prepared.X_train),
        )
        self.assertTrue(prepared.metadata["scaling"]["fit_on_training_only"])
        self.assertTrue(prepared.metadata["encoding"]["drop_first"])

    def test_stratified_split_preserves_label_ratio(self) -> None:
        prepared = prepare_data_from_frame(make_sample_data(), test_size=0.2)
        full_ratio = pd.concat([prepared.y_train, prepared.y_test]).mean()
        self.assertAlmostEqual(prepared.y_train.mean(), full_ratio, places=7)
        self.assertAlmostEqual(prepared.y_test.mean(), full_ratio, places=7)


if __name__ == "__main__":
    unittest.main()
