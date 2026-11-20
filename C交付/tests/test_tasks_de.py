import unittest

from tasks_de import DEPTHS, FOREST_SIZE, data_engineering, evaluate_models, fit_models


class TaskDETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepared = data_engineering.prepare_data()
        cls.models, cls.depths = fit_models(prepared.X_train, prepared.y_train)
        cls.metrics = evaluate_models(cls.models, prepared.X_test, prepared.y_test)
        cls.counts = prepared.y_test.value_counts()

    def test_baseline_predicts_no_churn(self):
        baseline = self.metrics.set_index("model").loc["全不流失基线"]
        self.assertEqual(baseline["tp"], 0)
        self.assertEqual(baseline["fp"], 0)
        self.assertEqual(baseline["fn"], self.counts[1])
        self.assertAlmostEqual(baseline["acc"], self.counts[0] / self.counts.sum())
        self.assertEqual(baseline["sens"], 0)
        self.assertEqual(baseline["spec"], 1)
        self.assertEqual(baseline["auc"], 0.5)

    def test_tree_depth_and_forest_configuration(self):
        self.assertEqual(self.depths["max_depth"].tolist(), DEPTHS)
        selected = self.depths.loc[self.depths["selected"]]
        self.assertEqual(selected["max_depth"].item(), self.models["决策树"].max_depth)
        self.assertEqual(selected["validation_auc_mean"].item(),
                         self.depths["validation_auc_mean"].max())
        self.assertEqual(len(self.models["随机森林"].estimators_), FOREST_SIZE)

    def test_all_models_score_the_same_holdout(self):
        self.assertTrue((self.metrics["tn"] + self.metrics["fp"]).eq(self.counts[0]).all())
        self.assertTrue((self.metrics["fn"] + self.metrics["tp"]).eq(self.counts[1]).all())
        scores = self.metrics[["acc", "sens", "spec", "auc"]]
        self.assertTrue((scores.ge(0) & scores.le(1)).all().all())
