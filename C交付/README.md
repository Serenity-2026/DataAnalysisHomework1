# 角色 C 交付

邓闽川负责任务三 D/E（基线、决策树和随机森林）、任务四（统一评估与加权对照）及报告。沿用 A 的 80/20 分层划分和预处理，B 的模型配置来自其交付，不改写 A、B 的训练代码。

- [任务三 D/E](任务三DE.ipynb)：四项指标、深度扫描、特征重要性，含运行输出。
- [任务四](任务四.ipynb)：全组 ROC、分层五折 AUC、加权对照和统一成绩单，含运行输出。
- **小组提交**：[小组实验 Notebook](小组实验.ipynb) 覆盖 A、B、C 的全部任务，含运行输出和分工；[小组实验报告](实验报告.md) 的第 5 节为三名成员的 AI 使用声明，走查记录区分技术核对与待邓闽川本人确认的事项。
- `outputs/`：与 Notebook 对应的汇总表和图。

## 复现

在 GitHub 分支签出后，将课程原始 CSV 放到仓库的 `data/telco_churn.csv`。原始 CSV 和 B 的逐行预测均不发布到远程。以 Python 3.13 和 scikit-learn 1.9.1 验证过本交付；低版本尤其可能使 MLP 重训结果不同。

```bash
python -m pip install -r requirements.txt jupyter 'scikit-learn==1.9.1'
cd C交付
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=240 小组实验.ipynb
python -m unittest discover -s tests -v
```

任务四优先使用本机 `B交付/B/outputs/b_models/` 中匹配的 B 交付预测；没有该目录时，读取 `b_model_config.json`，用原始课程 CSV 重训。复跑会刷新该目录下的图表和 CSV；报告引用的是本次集成验证原始 B 预测后的成绩。B 的原始交付使用 scikit-learn 1.4.2，因此不同环境下的重训指标不要求逐位一致。
