import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.svm import SVC

from task_four import B_MODELS, B_OUTPUT_DIR, build_estimator, class_weight_pair, load_b_config, load_b_delivery, run_evaluation
from tasks_de import data_engineering, positive_scores


class TaskFourTests(unittest.TestCase):
    def test_reproduction_configuration(self):
        self.assertEqual(tuple(load_b_config()), B_MODELS)

    @unittest.skipUnless(B_OUTPUT_DIR.is_dir(), "B 的原始逐条预测不随分支分发")
    def test_b_predictions_match_shared_holdout(self):
        prepared = data_engineering.prepare_data()
        models, scorecard, scores = load_b_delivery(prepared, B_OUTPUT_DIR)
        self.assertEqual(tuple(models), B_MODELS)
        for name, estimator in load_b_config().items():
            self.assertEqual(estimator.get_params(), models[name].get_params())
        self.assertTrue((scorecard["tn"] + scorecard["fp"]).eq(885).all())
        self.assertTrue((scorecard["tp"] + scorecard["fn"]).eq(318).all())
        self.assertTrue(all(values.index.equals(prepared.y_test.index) for values in scores.values()))

    def test_c_delivery_runs_without_b(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            scores, cv, comparison, changes, missing = run_evaluation(output_dir=output_dir)
            self.assertEqual(missing, list(B_MODELS))
            self.assertTrue(scores["auc"].is_monotonic_decreasing)
            self.assertEqual(set(cv["model"]), {"决策树", "随机森林"})
            self.assertTrue((cv.filter(like="fold_").ge(0) & cv.filter(like="fold_").le(1)).all().all())
            self.assertEqual(comparison["setting"].tolist(), ["默认", "加权"])
            self.assertEqual(comparison["tp"].diff().iloc[-1], changes["delta_tp"].item())
            for filename in ("fig07_roc_all.png", "fig08_imbalance.png", "scorecard.csv", "five_fold_auc.csv",
                             "class_weight_comparison.csv", "class_weight_changes.csv"):
                self.assertGreater((output_dir / filename).stat().st_size, 0)

    def test_svm_weighting_keeps_other_settings(self):
        features = np.arange(16).reshape(8, 2)
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        model = build_estimator("SVM-RBF核", SVC(C=2, random_state=42))
        self.assertIsNone(getattr(model.steps[0][1], "mean_", None))
        pair = class_weight_pair("SVM-RBF核", model, features, labels)
        default = pair["默认"].steps[-1][1].get_params()
        balanced = pair["加权"].steps[-1][1].get_params()
        self.assertIsNone(default.pop("class_weight"))
        self.assertEqual(balanced.pop("class_weight"), "balanced")
        self.assertEqual(default, balanced)
        np.testing.assert_allclose(positive_scores(pair["默认"], features), pair["默认"].decision_function(features))
