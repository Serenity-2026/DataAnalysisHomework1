"""作者：B 算法工程师。复用 A 的接口完成任务三 A/B/C。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import warnings
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.data_engineering import PreparedData, prepare_data
from src.data_engineering.config import DATA_PATH, PROJECT_ROOT, RANDOM_STATE


MODEL_NAMES = {
    "nb": "朴素贝叶斯",
    "svm_lin": "SVM-线性核",
    "svm_rbf": "SVM-RBF核",
    "mlp": "MLP",
}
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "b_models"


def report(name, y_true, y_pred, y_proba):
    """沿用 Notebook 的指标定义，正类 1 表示客户流失。"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    print(
        f"[{name}] Acc={acc:.4f}  灵敏度={sens:.4f}  "
        f"特异度={spec:.4f}  AUC={auc:.4f}",
        flush=True,
    )
    return dict(model=name, acc=acc, sens=sens, spec=spec, auc=auc)


def train_models(prepared: PreparedData) -> dict:
    """返回已拟合模型、成绩、测试预测和诊断，供 C 复用。"""
    models = {
        "nb": GaussianNB(),
        "svm_lin": SVC(
            kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE
        ),
        "svm_rbf": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            probability=True, random_state=RANDOM_STATE,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500,
            random_state=RANDOM_STATE,
        ),
    }
    results = []
    diagnostics = {}
    predictions = pd.DataFrame({"y_true": prepared.y_test})
    predictions.index.name = "source_row_index"

    for key, model in models.items():
        X_train = prepared.X_train_scaled
        X_test = prepared.X_test_scaled
        if key == "nb":
            X_train = prepared.X_train
            X_test = prepared.X_test

        print(f"开始训练 {MODEL_NAMES[key]}", flush=True)
        start = perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, prepared.y_train)
        elapsed = perf_counter() - start
        predicted = model.predict(X_test)
        probability = model.predict_proba(X_test)[:, 1]
        results.append(report(MODEL_NAMES[key], prepared.y_test, predicted, probability))
        predictions[f"{key}_pred"] = predicted
        predictions[f"{key}_proba"] = probability
        diagnostics[key] = {
            "fit_seconds": elapsed,
            "parameters": model.get_params(),
            "warnings": [str(item.message) for item in caught],
            "confusion_matrix_labels_0_1": confusion_matrix(
                prepared.y_test, predicted, labels=[0, 1]
            ).tolist(),
        }

    for key in ("svm_lin", "svm_rbf"):
        model = models[key]
        diagnostics[key]["support_vectors_by_class"] = {
            str(int(label)): int(count)
            for label, count in zip(model.classes_, model.n_support_)
        }
        diagnostics[key]["support_vectors_total"] = int(model.n_support_.sum())
        print(
            f"{MODEL_NAMES[key]} 各类别支持向量数 {model.n_support_.tolist()}，"
            f"总数 {model.n_support_.sum()}", flush=True,
        )

    mlp = models["mlp"]
    diagnostics["mlp"].update({
        "n_iter": int(mlp.n_iter_),
        "final_training_loss": float(mlp.loss_),
        "last_10_losses": [float(value) for value in mlp.loss_curve_[-10:]],
        "reached_iteration_limit": bool(mlp.n_iter_ == mlp.max_iter),
    })
    metrics = pd.DataFrame(results)
    diagnostics["rbf_minus_linear_auc"] = float(
        results[2]["auc"] - results[1]["auc"]
    )
    print(f"MLP 实际迭代 {mlp.n_iter_} 轮，最终训练损失 {mlp.loss_:.6f}", flush=True)
    return {
        "models": models,
        "results": metrics,
        "predictions": predictions,
        "diagnostics": diagnostics,
    }


def save_outputs(experiment: dict, prepared: PreparedData, data_path: Path, output_dir: Path):
    """训练全部成功后保存到临时目录，再整体发布本次实验产出。"""
    if output_dir.exists():
        raise FileExistsError(f"输出目录已存在，请用 --output-dir 指定新目录：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent, prefix="b_run_") as scratch:
        destination = Path(scratch) / "completed"
        destination.mkdir()
        experiment["results"].to_csv(destination / "metrics.csv", index=False, encoding="utf-8-sig")
        experiment["predictions"].to_csv(destination / "test_predictions.csv", encoding="utf-8-sig")
        metadata = {
            "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            "split": prepared.metadata["split"],
            "numeric_cleaning": prepared.metadata["numeric_cleaning"],
            "feature_names": prepared.feature_names,
            "train_row_indices": prepared.X_train.index.tolist(),
            "test_row_indices": prepared.X_test.index.tolist(),
            "versions": {
                "python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            },
            "diagnostics": experiment["diagnostics"],
        }
        (destination / "experiment_details.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        losses = experiment["models"]["mlp"].loss_curve_
        pd.DataFrame({"epoch": range(1, len(losses) + 1), "loss": losses}).to_csv(
            destination / "mlp_loss.csv", index=False
        )
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        preferred_fonts = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
        fonts = [name for name in preferred_fonts if name in available_fonts]
        with plt.rc_context({"font.sans-serif": fonts or ["DejaVu Sans"], "axes.unicode_minus": False}):
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(range(1, len(losses) + 1), losses)
            axis.set_xlabel("训练轮数" if fonts else "Epoch")
            axis.set_ylabel("训练损失" if fonts else "Training loss")
            axis.set_title("MLP 训练损失曲线" if fonts else "MLP training loss")
            figure.tight_layout()
            figure.savefig(destination / "fig04_mlp_loss.png", dpi=150)
            plt.close(figure)
        destination.rename(output_dir)


def main():
    parser = argparse.ArgumentParser(description="B 算法工程师：NB、SVM 与 MLP")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("输出目录已存在，请用 --output-dir 指定新的实验目录。")
    prepared = prepare_data(args.data)
    print(
        f"沿用 A 的数据接口：训练 {prepared.X_train.shape}，测试 {prepared.X_test.shape}",
        flush=True,
    )
    experiment = train_models(prepared)
    save_outputs(experiment, prepared, args.data, args.output_dir.resolve())
    print(f"实验产出已保存至 {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
