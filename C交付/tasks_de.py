"""角色 C（邓闽川）：任务三 D/E，树模型与全不流失基线。"""

import sys
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

DELIVERY_DIR = Path(__file__).resolve().parent
A_DELIVERY_DIR = DELIVERY_DIR.parent / "A交付"
A_PROJECT_DIR = A_DELIVERY_DIR if (A_DELIVERY_DIR / "src" / "data_engineering").is_dir() else DELIVERY_DIR.parent
if str(A_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(A_PROJECT_DIR))
data_engineering = import_module("src.data_engineering")


OUTPUT_DIR = DELIVERY_DIR / "outputs" / "tasks_de"
RANDOM_STATE = data_engineering.RANDOM_STATE
DEPTHS = list(range(2, 16))
FOREST_SIZE = 200


def tree_depth_search():
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"max_depth": DEPTHS},
        scoring="roc_auc",
        cv=splitter,
        return_train_score=True,
        error_score="raise",
    )


def fit_models(X_train, y_train):
    """只用 A 的未缩放训练集训练，按分层五折平均 AUC 选树深度。"""
    search = tree_depth_search().fit(X_train, y_train)

    depths = pd.DataFrame({
        "max_depth": DEPTHS,
        "train_auc_mean": search.cv_results_["mean_train_score"],
        "validation_auc_mean": search.cv_results_["mean_test_score"],
        "validation_auc_std": search.cv_results_["std_test_score"],
    })
    depths["selected"] = depths["max_depth"].eq(search.best_params_["max_depth"])
    models = {
        "全不流失基线": DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train),
        "决策树": search.best_estimator_,
        "随机森林": RandomForestClassifier(
            n_estimators=FOREST_SIZE, random_state=RANDOM_STATE, n_jobs=-1
        ).fit(X_train, y_train),
    }
    return models, depths


def positive_scores(model, features):
    positive_index = np.flatnonzero(model.classes_ == 1).item()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, positive_index]
    margins = model.decision_function(features)
    return margins if positive_index == 1 else -margins


def evaluate_models(models, X_test, y_test):
    """在共同的留出集上计算阈值指标和不依赖阈值的 AUC。"""
    return pd.DataFrame([
        score_predictions(name, y_test, model.predict(X_test), positive_scores(model, X_test))
        for name, model in models.items()
    ])


def score_predictions(name, labels, predictions, scores):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "model": name,
        "acc": accuracy_score(labels, predictions),
        "sens": tp / (tp + fn),
        "spec": tn / (tn + fp),
        "auc": roc_auc_score(labels, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def save_figures(depths, importance, output_dir):
    selected = depths.loc[depths["selected"]].iloc[0]
    figure, axes = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axes.plot(depths["max_depth"], depths["train_auc_mean"], "o-", label="Training folds")
    axes.errorbar(
        depths["max_depth"], depths["validation_auc_mean"],
        yerr=depths["validation_auc_std"], fmt="s-", capsize=3,
        label="Validation folds (mean ± SD)",
    )
    axes.axvline(selected["max_depth"], color="0.4", linestyle="--",
                 label=f"Selected depth = {int(selected['max_depth'])}")
    axes.set(xlabel="Maximum depth", ylabel="ROC AUC", title="Decision tree depth scan",
             xticks=DEPTHS)
    axes.legend()
    figure.savefig(output_dir / "fig05_tree_depth.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(8, 5), layout="constrained")
    top = importance.head(10).iloc[::-1]
    axes.barh(top.index, top.values)
    axes.set(xlabel="Mean decrease in impurity", title="Random forest: top 10 features")
    figure.savefig(output_dir / "fig06_rf_importance.png", dpi=150)
    plt.close(figure)


def run_experiment(output_dir=OUTPUT_DIR):
    """复用 A 的划分，保存 D/E 的表格与图，并返回模型供任务四使用。"""
    prepared = data_engineering.prepare_data()
    models, depths = fit_models(prepared.X_train, prepared.y_train)
    metrics = evaluate_models(models, prepared.X_test, prepared.y_test)
    importance = pd.Series(
        models["随机森林"].feature_importances_,
        index=pd.Index(prepared.feature_names, name="feature"),
        name="importance",
    ).sort_values(ascending=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    depths.to_csv(output_dir / "tree_depth_scan.csv", index=False)
    importance.to_csv(output_dir / "rf_feature_importance.csv")
    save_figures(depths, importance, output_dir)
    return models, metrics, depths, importance


def main():
    models, metrics, depths, importance = run_experiment()
    print(metrics.round(4).to_string(index=False))
    print(f"\n最优树深度（训练集五折）：{models['决策树'].max_depth}")
    print(f"验证 AUC：{depths.loc[depths['selected'], 'validation_auc_mean'].item():.4f}")
    print(f"\n特征重要性 Top 3：\n{importance.head(3).round(4).to_string()}")
    print(f"\n产出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
