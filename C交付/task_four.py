"""角色 C（邓闽川）：统一评分、ROC、五折验证与类别加权对照。"""

import json
from hashlib import sha256

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tasks_de import (
    DELIVERY_DIR,
    RANDOM_STATE,
    data_engineering,
    evaluate_models,
    fit_models,
    positive_scores,
    score_predictions,
    tree_depth_search,
)


OUTPUT_DIR = DELIVERY_DIR / "outputs" / "task_four"
B_OUTPUT_DIR = DELIVERY_DIR.parent / "B交付" / "B" / "outputs" / "b_models"
B_CONFIG_PATH = DELIVERY_DIR / "b_model_config.json"
BASELINE = "全不流失基线"
MODEL_CONFIGS = {
    "朴素贝叶斯": ("nb", GaussianNB),
    "SVM-线性核": ("svm_lin", SVC),
    "SVM-RBF核": ("svm_rbf", SVC),
    "MLP": ("mlp", MLPClassifier),
}
B_MODELS = tuple(MODEL_CONFIGS)
SCALED_MODELS = {"SVM-线性核", "SVM-RBF核", "MLP"}
WEIGHTED_MODELS = ("随机森林", "SVM-RBF核")
NAMES = {"决策树": "Decision tree", "随机森林": "Random forest", "朴素贝叶斯": "Gaussian NB",
         "SVM-线性核": "Linear SVM", "SVM-RBF核": "RBF SVM", "MLP": "MLP"}


def build_estimator(name, estimator):
    estimator = clone(estimator)
    return make_pipeline(StandardScaler(), estimator) if name in SCALED_MODELS else estimator


def load_b_config():
    configuration = json.loads(B_CONFIG_PATH.read_text(encoding="utf-8"))
    if sha256(data_engineering.DATA_PATH.read_bytes()).hexdigest() != configuration["data_sha256"]:
        raise ValueError("当前 CSV 与小组实验数据的哈希不一致")
    return {name: constructor(**configuration["parameters"][key])
            for name, (key, constructor) in MODEL_CONFIGS.items()}


def load_b_delivery(prepared, source_dir=B_OUTPUT_DIR):
    details = json.loads((source_dir / "experiment_details.json").read_text(encoding="utf-8"))
    expected = {
        "data_sha256": sha256(data_engineering.DATA_PATH.read_bytes()).hexdigest(),
        "split": prepared.metadata["split"],
        "numeric_cleaning": prepared.metadata["numeric_cleaning"],
        "feature_names": prepared.feature_names,
        "train_row_indices": prepared.y_train.index.tolist(),
        "test_row_indices": prepared.y_test.index.tolist(),
    }
    for name, value in expected.items():
        if details[name] != value:
            raise ValueError(f"B 的交付与 A 的预处理不一致：{name}")

    predictions = pd.read_csv(source_dir / "test_predictions.csv", index_col="source_row_index")
    if not predictions.index.equals(prepared.y_test.index) or not np.array_equal(
        predictions["y_true"], prepared.y_test
    ):
        raise ValueError("B 交付的测试行索引或真实标签与 A 不一致")

    models = {}
    rows = []
    scores = {}
    for name, (key, constructor) in MODEL_CONFIGS.items():
        models[name] = constructor(**details["diagnostics"][key]["parameters"])
        guesses = predictions[f"{key}_pred"]
        scores[name] = predictions[f"{key}_proba"]
        if not guesses.isin([0, 1]).all() or not scores[name].between(0, 1).all():
            raise ValueError(f"B 交付的测试标签或预测概率无效：{name}")
        rows.append(score_predictions(name, prepared.y_test, guesses, scores[name]))

    scorecard = pd.DataFrame(rows).set_index("model")
    reported = pd.read_csv(source_dir / "metrics.csv", index_col="model")
    if set(reported.index) != set(models) or not np.allclose(
        reported.reindex(scorecard.index)[["acc", "sens", "spec", "auc"]],
        scorecard[["acc", "sens", "spec", "auc"]], atol=1e-10, rtol=0,
    ):
        raise ValueError("B 的指标表与逐行预测重算结果不一致")
    return models, scorecard.reset_index(), scores


def positive_auc(model, features, labels):
    return roc_auc_score(labels, positive_scores(model, features))


def five_fold_auc(models, X_train, y_train):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 1)
    fold_scores = {}
    for name, model in models.items():
        if name == BASELINE:
            continue
        estimator = tree_depth_search() if name == "决策树" else clone(model)
        fold_scores[name] = cross_val_score(
            estimator, X_train, y_train, scoring=positive_auc, cv=splitter, error_score="raise"
        )
    scores = pd.DataFrame.from_dict(fold_scores, orient="index", columns=[f"fold_{fold}" for fold in range(1, 6)])
    scores["mean_auc"] = scores.mean(axis=1)
    scores["std_auc"] = scores.iloc[:, :5].std(axis=1, ddof=0)
    return scores.rename_axis("model").reset_index().sort_values("mean_auc", ascending=False)


def class_weight_pair(name, model, X_train, y_train):
    if name in SCALED_MODELS:
        parameter = f"{model.steps[-1][0]}__class_weight"
    else:
        parameter = "class_weight"
    return {
        label: clone(model).set_params(**{parameter: weight}).fit(X_train, y_train)
        for label, weight in (("默认", None), ("加权", "balanced"))
    }


def compare_weights(models, prepared):
    comparisons = []
    changes = []
    for name in WEIGHTED_MODELS:
        if name not in models:
            continue
        pair = class_weight_pair(name, models[name], prepared.X_train, prepared.y_train)
        scores = evaluate_models(pair, prepared.X_test, prepared.y_test).rename(columns={"model": "setting"})
        scores.insert(0, "model", name)
        before, after = scores.iloc[0], scores.iloc[1]
        changes.append({
            "model": name,
            "delta_sensitivity_pp": 100 * (after["sens"] - before["sens"]),
            "delta_specificity_pp": 100 * (after["spec"] - before["spec"]),
            "delta_tp": after["tp"] - before["tp"],
            "delta_fp": after["fp"] - before["fp"],
        })
        comparisons.append(scores)
    return pd.concat(comparisons, ignore_index=True), pd.DataFrame(changes)


def save_roc(test_scores, scorecard, labels, path):
    figure, axes = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for row in scorecard.itertuples(index=False):
        if row.model == BASELINE:
            continue
        fpr, tpr, _ = roc_curve(labels, test_scores[row.model])
        axes.plot(fpr, tpr, label=f"{NAMES[row.model]} (AUC = {row.auc:.4f})")
    axes.plot([0, 1], [0, 1], "k--", label="Chance / constant baseline (AUC = 0.5000)")
    model_count = sum(scorecard["model"].ne(BASELINE))
    axes.set(xlabel="False positive rate (1 - specificity)", ylabel="True positive rate (sensitivity)",
             title=f"Holdout ROC: {model_count} trained models")
    axes.legend(loc="lower right")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_weight_plot(comparison, path):
    names = comparison["model"].unique()
    figure, panels = plt.subplots(1, len(names), figsize=(5 * len(names), 4), layout="constrained", squeeze=False)
    for name, axes in zip(names, panels[0]):
        scores = comparison.loc[comparison["model"].eq(name), ["sens", "spec"]].T
        scores.index = ["Sensitivity", "Specificity"]
        scores.columns = ["Default", "Balanced"]
        scores.plot.bar(ax=axes, rot=0, ylim=(0, 1.1), color=["#4274a7", "#d27b32"])
        for bars in axes.containers:
            axes.bar_label(bars, fmt="%.3f", padding=2)
        axes.set(title=NAMES[name], ylabel="Rate")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def run_evaluation(b_source=None, output_dir=OUTPUT_DIR, *, refit_b=False):
    """归并共享测试集的成绩；在本地复跑交叉验证和受控加权对照。"""
    prepared = data_engineering.prepare_data()
    models, _ = fit_models(prepared.X_train, prepared.y_train)
    scorecard = evaluate_models(models, prepared.X_test, prepared.y_test)
    test_scores = {
        name: positive_scores(model, prepared.X_test) for name, model in models.items() if name != BASELINE
    }
    if b_source is not None:
        estimators, b_scorecard, b_scores = load_b_delivery(prepared, b_source)
        models.update({name: build_estimator(name, estimator) for name, estimator in estimators.items()})
        scorecard = pd.concat([scorecard, b_scorecard], ignore_index=True)
        test_scores.update(b_scores)
    elif refit_b:
        estimators = load_b_config()
        trained = {name: build_estimator(name, estimator).fit(prepared.X_train, prepared.y_train)
                   for name, estimator in estimators.items()}
        models.update(trained)
        scorecard = pd.concat([scorecard, evaluate_models(trained, prepared.X_test, prepared.y_test)],
                              ignore_index=True)
        test_scores.update({name: positive_scores(model, prepared.X_test) for name, model in trained.items()})

    scorecard = scorecard.sort_values("auc", ascending=False)
    cv = five_fold_auc(models, prepared.X_train, prepared.y_train)
    comparison, changes = compare_weights(models, prepared)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(output_dir / "scorecard.csv", index=False)
    cv.to_csv(output_dir / "five_fold_auc.csv", index=False)
    comparison.to_csv(output_dir / "class_weight_comparison.csv", index=False)
    changes.to_csv(output_dir / "class_weight_changes.csv", index=False)
    save_roc(test_scores, scorecard, prepared.y_test, output_dir / "fig07_roc_all.png")
    save_weight_plot(comparison, output_dir / "fig08_imbalance.png")
    missing = [name for name in B_MODELS if name not in models]
    return scorecard, cv, comparison, changes, missing
