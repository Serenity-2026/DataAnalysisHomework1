"""任务一：生成真实数据质量统计、EDA 图表和摘要。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import DATA_PATH, EDA_OUTPUT_DIR
from .quality import build_quality_summary, load_raw_data


PLOT_FILENAMES = [
    "01_churn_distribution.png",
    "02_contract_churn_rate.png",
    "03_tenure_by_churn.png",
    "04_monthly_charges_by_churn.png",
    "05_internet_service_churn_rate.png",
    "06_feature_correlation_heatmap.png",
]


@dataclass
class EDAResult:
    """EDA 运行结果及生成文件。"""

    quality_summary: dict[str, Any]
    analytical_rows: int
    churn_counts: dict[str, int]
    churn_ratios: dict[str, float]
    top_correlations: list[dict[str, Any]]
    output_paths: list[Path]


def _prepare_analysis_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    """构造仅用于描述性 EDA 的分析表，不修改原始数据。"""

    analysis = raw_data.copy(deep=True)
    for column in analysis.select_dtypes(include=["object", "string"]).columns:
        stripped = analysis[column].astype("string").str.strip()
        analysis[column] = stripped.mask(stripped.eq(""), pd.NA)

    # 缺失标签的截断行不能参与按流失标签分组的描述性分析。
    analysis = analysis.loc[analysis["Churn"].notna()].copy()
    labels = set(analysis["Churn"].dropna().unique())
    if labels - {"Yes", "No"}:
        raise ValueError(f"Churn 存在未识别标签：{sorted(labels - {'Yes', 'No'})}")

    analysis["TotalCharges"] = pd.to_numeric(
        analysis["TotalCharges"], errors="coerce"
    )
    tenure_zero_missing = analysis["TotalCharges"].isna() & analysis["tenure"].eq(0)
    analysis.loc[tenure_zero_missing, "TotalCharges"] = 0.0
    analysis["ChurnBinary"] = analysis["Churn"].map({"No": 0, "Yes": 1})
    return analysis


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_churn_distribution(data: pd.DataFrame, path: Path) -> None:
    counts = data["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    ratios = counts / counts.sum()
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    bars = axis.bar(counts.index, counts.values, color=["#4C78A8", "#E45756"])
    axis.set_title("Customer Churn Distribution")
    axis.set_xlabel("Churn label")
    axis.set_ylabel("Customer count")
    axis.set_ylim(0, max(counts.values) * 1.15)
    for bar, count, ratio in zip(bars, counts.values, ratios.values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts.values) * 0.02,
            f"{count:,}\n{ratio:.2%}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_rate_by_category(
    data: pd.DataFrame,
    category: str,
    path: Path,
    *,
    title: str,
    xlabel: str,
) -> pd.DataFrame:
    grouped = (
        data.groupby(category, observed=True)["ChurnBinary"]
        .agg(churn_rate="mean", customer_count="size")
        .sort_values("churn_rate", ascending=False)
    )
    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    bars = axis.bar(
        grouped.index.astype(str), grouped["churn_rate"], color="#F58518"
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Churn rate")
    axis.set_ylim(0, min(1.0, float(grouped["churn_rate"].max()) * 1.25 + 0.02))
    axis.tick_params(axis="x", rotation=18)
    for bar, rate, count in zip(
        bars, grouped["churn_rate"], grouped["customer_count"]
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{rate:.1%}\nn={int(count):,}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    figure.tight_layout()
    _save_figure(figure, path)
    return grouped


def _plot_numeric_by_churn(
    data: pd.DataFrame,
    feature: str,
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    sns.boxplot(
        data=data,
        x="Churn",
        y=feature,
        hue="Churn",
        order=["No", "Yes"],
        hue_order=["No", "Yes"],
        palette={"No": "#4C78A8", "Yes": "#E45756"},
        legend=False,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Churn label")
    axis.set_ylabel(ylabel)
    figure.tight_layout()
    _save_figure(figure, path)


def _correlation_inputs(data: pd.DataFrame) -> pd.DataFrame:
    features = data.drop(columns=["customerID", "Churn"]).copy()
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].fillna(features[column].median())
        else:
            mode = features[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "<MISSING>"
            features[column] = features[column].fillna(fill_value)
    return pd.get_dummies(features, drop_first=True, dtype=float)


def _top_predictor_correlations(
    correlation: pd.DataFrame, *, limit: int = 5
) -> list[dict[str, Any]]:
    predictors = correlation.drop(index="ChurnBinary", columns="ChurnBinary")
    upper_mask = np.triu(np.ones(predictors.shape, dtype=bool), k=1)
    pairs = predictors.where(upper_mask).stack().rename("correlation")
    if pairs.empty:
        return []
    ordered = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(limit)
    return [
        {
            "feature_1": str(feature_1),
            "feature_2": str(feature_2),
            "correlation": float(value),
        }
        for (feature_1, feature_2), value in ordered.items()
    ]


def _plot_correlation_heatmap(
    encoded: pd.DataFrame, path: Path
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    correlation = encoded.corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(22, 18))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        linewidths=0.1,
        cbar_kws={"label": "Pearson correlation"},
        ax=axis,
    )
    axis.set_title(
        "Feature Correlation Heatmap\n"
        "Numeric features plus drop-first one-hot categorical features"
    )
    axis.tick_params(axis="x", labelsize=7, rotation=90)
    axis.tick_params(axis="y", labelsize=7, rotation=0)
    figure.tight_layout()
    _save_figure(figure, path)
    return correlation, _top_predictor_correlations(correlation)


def _build_summary_markdown(
    quality: dict[str, Any],
    analysis: pd.DataFrame,
    contract_rates: pd.DataFrame,
    internet_rates: pd.DataFrame,
    top_correlations: list[dict[str, Any]],
) -> str:
    churn_counts = analysis["Churn"].value_counts().reindex(["No", "Yes"])
    churn_ratios = churn_counts / churn_counts.sum()
    total = int(churn_counts.sum())

    highest_contract = contract_rates.iloc[0]
    highest_contract_name = str(contract_rates.index[0])
    highest_internet = internet_rates.iloc[0]
    highest_internet_name = str(internet_rates.index[0])

    lines = [
        "# A 数据工程师 EDA 摘要",
        "",
        "## 数据与质量检查",
        "",
        f"- 原始数据：{quality['shape']['rows']} 行、{quality['shape']['columns']} 列。",
        f"- 普通 `isna()` 检出的缺失：最后一条截断记录除 `customerID` 和残缺的 `gender` 外有 19 个字段缺失；缺失标签行共 {quality['missing_target_rows']} 条。",
        f"- 全字段均为空的记录：{quality['all_empty_rows']} 条；严重不完整记录：{quality['severely_incomplete_rows']} 条。",
        f"- 完全重复行：{quality['fully_duplicated_rows']} 条；重复 `customerID`：{quality['customer_id']['duplicated_non_missing']} 条。",
        f"- `TotalCharges` 原始类型：`{quality['total_charges']['original_dtype']}`；纯空白字符串 {quality['total_charges']['blank_strings']} 条；非空文本转换失败 {quality['total_charges']['nonblank_conversion_failures']} 条。",
        f"- `TotalCharges` 数值转换后缺失 {quality['total_charges']['numeric_missing_after_conversion']} 条，其中 tenure=0 为 {quality['total_charges']['numeric_missing_with_tenure_zero']} 条，另有 {quality['total_charges']['numeric_missing_with_tenure_nonzero']} 条来自截断记录。",
        "",
        "## 标签分布",
        "",
        f"剔除缺少标签的截断记录后，描述性分析使用 {total} 条客户记录。",
        f"`No` 为 {int(churn_counts['No'])} 条（{churn_ratios['No']:.2%}），`Yes` 为 {int(churn_counts['Yes'])} 条（{churn_ratios['Yes']:.2%}）。",
        "标签存在明显数量差异，后续模型评估不能只看准确率，还应结合灵敏度、特异度和 AUC；本模块不训练或比较模型。",
        "",
        "## 特征观察",
        "",
        f"- `Contract` 中流失率最高的观测组为 `{highest_contract_name}`（{float(highest_contract['churn_rate']):.2%}，n={int(highest_contract['customer_count'])}）。",
        f"- `InternetService` 中流失率最高的观测组为 `{highest_internet_name}`（{float(highest_internet['churn_rate']):.2%}，n={int(highest_internet['customer_count'])}）。",
        "- tenure 与 MonthlyCharges 的箱线图按真实 Churn 标签展示分布差异；这些差异是描述性关联，不能解释为因果关系。",
        "",
        "## 相关性热力图",
        "",
        "热力图排除 `customerID`，纳入数值特征、二元 Churn 标签，以及对类别特征执行 `drop_first=True` 后的独热编码列。该编码仅用于全量描述性相关分析，不用于模型训练/测试划分。",
    ]
    if top_correlations:
        strongest = top_correlations[0]
        lines.extend(
            [
                f"绝对相关系数最高的预测特征对为 `{strongest['feature_1']}` 与 `{strongest['feature_2']}`（r={strongest['correlation']:.3f}）。",
                "这说明 GaussianNB 的条件独立假设需要结合这些相关特征进一步讨论，但不能仅凭相关系数断言模型表现。",
                "",
                "绝对相关系数较高的前五组预测特征：",
                "",
            ]
        )
        for pair in top_correlations:
            lines.append(
                f"- `{pair['feature_1']}` / `{pair['feature_2']}`：r={pair['correlation']:.3f}"
            )
    lines.extend(
        [
            "",
            "## 图表文件",
            "",
            *[f"- `{filename}`" for filename in PLOT_FILENAMES],
            "",
        ]
    )
    return "\n".join(lines)


def _assert_output_targets(paths: list[Path], *, force: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"输出已存在：{joined}；如需重新生成请传入 --force。")


def run_eda(
    data_path: str | Path = DATA_PATH,
    output_dir: str | Path = EDA_OUTPUT_DIR,
    *,
    force: bool = False,
) -> EDAResult:
    """运行数据质量检查和全部必需 EDA，并保存可追溯产出。"""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    plot_paths = [destination / filename for filename in PLOT_FILENAMES]
    quality_path = destination / "data_quality_summary.json"
    summary_path = destination / "eda_summary.md"
    log_path = destination / "eda_run.log"
    output_paths = [*plot_paths, quality_path, summary_path, log_path]
    _assert_output_targets(output_paths, force=force)

    raw_data = load_raw_data(data_path)
    quality = build_quality_summary(raw_data)
    analysis = _prepare_analysis_frame(raw_data)

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_churn_distribution(analysis, plot_paths[0])
    contract_rates = _plot_rate_by_category(
        analysis,
        "Contract",
        plot_paths[1],
        title="Churn Rate by Contract Type",
        xlabel="Contract type",
    )
    _plot_numeric_by_churn(
        analysis,
        "tenure",
        plot_paths[2],
        title="Tenure Distribution by Churn Label",
        ylabel="Tenure (months)",
    )
    _plot_numeric_by_churn(
        analysis,
        "MonthlyCharges",
        plot_paths[3],
        title="Monthly Charges Distribution by Churn Label",
        ylabel="Monthly charges",
    )
    internet_rates = _plot_rate_by_category(
        analysis,
        "InternetService",
        plot_paths[4],
        title="Churn Rate by Internet Service",
        xlabel="Internet service",
    )

    encoded = _correlation_inputs(analysis)
    _, top_correlations = _plot_correlation_heatmap(encoded, plot_paths[5])

    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        _build_summary_markdown(
            quality, analysis, contract_rates, internet_rates, top_correlations
        ),
        encoding="utf-8",
    )

    churn_counts_series = analysis["Churn"].value_counts().reindex(["No", "Yes"])
    churn_ratios_series = churn_counts_series / churn_counts_series.sum()
    churn_counts = {
        str(label): int(count) for label, count in churn_counts_series.items()
    }
    churn_ratios = {
        str(label): float(ratio) for label, ratio in churn_ratios_series.items()
    }
    log_lines = [
        "A 数据工程师任务一真实运行记录",
        f"输入文件: {Path(data_path)}",
        f"原始形状: {raw_data.shape}",
        f"描述性分析样本数: {len(analysis)}",
        f"Churn 计数: {churn_counts}",
        f"Churn 比例: {churn_ratios}",
        f"全空行: {quality['all_empty_rows']}",
        f"严重不完整行: {quality['severely_incomplete_rows']}",
        f"TotalCharges 空白字符串: {quality['total_charges']['blank_strings']}",
        f"已生成图表数: {len(plot_paths)}",
    ]
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return EDAResult(
        quality_summary=quality,
        analytical_rows=int(len(analysis)),
        churn_counts=churn_counts,
        churn_ratios=churn_ratios,
        top_correlations=top_correlations,
        output_paths=output_paths,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行客户流失数据 EDA")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="原始 CSV 路径")
    parser.add_argument(
        "--output-dir", type=Path, default=EDA_OUTPUT_DIR, help="EDA 输出目录"
    )
    parser.add_argument(
        "--force", action="store_true", help="覆盖由本脚本生成的同名输出"
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_eda(args.data, args.output_dir, force=args.force)
    print("EDA 完成。")
    print(f"分析样本数: {result.analytical_rows}")
    print(f"Churn 计数: {result.churn_counts}")
    print(f"Churn 比例: {result.churn_ratios}")
    for path in result.output_paths:
        print(f"已生成: {path}")


if __name__ == "__main__":
    main()
