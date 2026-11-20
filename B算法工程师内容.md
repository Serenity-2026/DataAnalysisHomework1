# B 算法工程师交付说明

作者为 B 算法工程师。此交付覆盖任务三 3.A、3.B、3.C，使用 A 的 `prepare_data()` 生成数据。正式实验采用 A 当前默认的 20% 分层测试集、随机种子 42、训练集统计量填补规则，以及 tenure=0 时累计费用缺失填 0 的规则。

## 运行

在仓库根目录打开 PowerShell，使用当前已验证的 Python 环境。

```powershell
$env:PYTHONNOUSERSITE = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
python -m src.algorithm_engineering.run_models
```

当前已有正式结果。重复实验时指定新的输出目录，例如下列命令。目录已存在时程序会要求选择新目录，以便保留每次实验记录。

```powershell
python -m src.algorithm_engineering.run_models --output-dir outputs/b_models_repeat
```

训练集 4810 条、测试集 1203 条，特征数 30。测试集包含未流失客户 885 条、流失客户 318 条。四个模型共享相同的数据划分。GaussianNB 使用未缩放特征，SVM 和 MLP 使用 A 提供的缩放特征。

## 实际结果

下表来自 `outputs/b_models/metrics.csv`。正类为流失客户，AUC 使用正类预测概率计算，灵敏度与特异度使用模型的 `predict()` 结果计算。

| 模型 | 准确率 | 灵敏度 | 特异度 | AUC |
|---|---|---|---|---|
| 朴素贝叶斯 | 0.6492 | 0.8679 | 0.5706 | 0.8067 |
| SVM 线性核 | 0.7938 | 0.5472 | 0.8825 | 0.8199 |
| SVM RBF 核 | 0.7955 | 0.5000 | 0.9017 | 0.8029 |
| MLP | 0.7598 | 0.5503 | 0.8350 | 0.7859 |

| 模型 | TN | FP | FN | TP |
|---|---|---|---|---|
| 朴素贝叶斯 | 505 | 380 | 42 | 276 |
| SVM 线性核 | 781 | 104 | 144 | 174 |
| SVM RBF 核 | 798 | 87 | 159 | 159 |
| MLP | 739 | 146 | 143 | 175 |

线性核 SVM 在本次四个模型中 AUC 最高。NB 识别出 276 名流失客户，也将 380 名未流失客户判为流失，因此业务上需要同时考虑挽留覆盖率和联系成本。最终综合比较还需结合 C 的树模型、统一评估及交叉验证。

## 条件独立假设分析依据

在 A 返回的训练集上，`tenure` 与 `TotalCharges` 的相关系数约为 0.8200。按训练标签分组后，未流失组为 0.7886，流失组为 0.9507。此外，`PhoneService_Yes` 与 `MultipleLines_No phone service` 在两组内的相关系数均约为 -1。条件独立要求给定类别后特征相互独立，这些组内关联说明假设受到违背。

GaussianNB 对相关证据的重复利用可能影响预测概率和分类结果。实际观测到的高灵敏度及低特异度需要结合混淆矩阵解释，相关性本身不能证明这组性能完全由条件独立假设违背造成。组内相关系数可按下列方式复核。

```python
from src.data_engineering import prepare_data

prepared = prepare_data()
for label in (0, 1):
    group = prepared.X_train.loc[prepared.y_train.eq(label)]
    print(label, group["tenure"].corr(group["TotalCharges"]))
    print(label, group["PhoneService_Yes"].corr(
        group["MultipleLines_No phone service"]
    ))
```

## SVM 核函数对照

两个 SVC 均采用 `C=1.0`、`probability=True`、`random_state=42`，RBF 使用 `gamma='scale'`，类别权重均为默认值。

| 核函数 | 未流失类支持向量数 | 流失类支持向量数 | 总数 |
|---|---|---|---|
| 线性核 | 1095 | 1088 | 2183 |
| RBF 核 | 1248 | 1050 | 2298 |

RBF 相对线性核的 AUC 差值为 -0.01695。RBF 核相当于在高维特征空间中构建分类边界，可以表示原空间中的非线性关系。本次结果说明默认参数下的非线性模型没有带来 AUC 提升。正则化强度、核宽度与数据关系都可能影响表现，后续若调参应使用训练集内部验证。

## MLP 训练过程

初始配置由 B 阶段选定，使用两层隐层 `(64, 32)`、ReLU 激活、Adam 优化器、最多 500 轮、随机种子 42，其余参数使用默认值。完整参数和软件版本记录在 `experiment_details.json`。

实际训练 298 轮，训练损失从 0.560335 降至最终的 0.184104，最低观测值为 0.171147。没有出现 `ConvergenceWarning`，训练在达到最大轮数前因损失改善不足而停止。曲线整体下降，尾部存在波动，可以说明满足了本次优化的停止条件。复核得到训练 AUC 为 0.9776，测试 AUC 为 0.7859，存在明显过拟合迹象。泛化表现仍应依据验证数据和交叉验证，继续增加轮数也需要观察验证结果。

损失曲线为 `outputs/b_models/fig04_mlp_loss.png`，逐轮数据为 `mlp_loss.csv`。单层感知机只能构造线性边界，XOR 的两类无法用一条直线分开；加入带非线性激活的隐层后，网络可以组合出非线性决策边界。

## 给 C 的接口

在仓库根目录运行以下代码可以得到已拟合模型及 B 的成绩单。这会重新训练四个模型，数据划分由 A 统一提供。

```python
from src.data_engineering import prepare_data
from src.algorithm_engineering.run_models import train_models

prepared = prepare_data()
experiment = train_models(prepared)
nb = experiment["models"]["nb"]
svm_lin = experiment["models"]["svm_lin"]
svm_rbf = experiment["models"]["svm_rbf"]
mlp = experiment["models"]["mlp"]
results = experiment["results"].to_dict("records")

X_train, X_test = prepared.X_train, prepared.X_test
X_train_s, X_test_s = prepared.X_train_scaled, prepared.X_test_scaled
y_train, y_test = prepared.y_train, prepared.y_test
```

C 也可以直接读取 `outputs/b_models/test_predictions.csv` 绘制本次测试集 ROC，其中 `y_true` 是真实标签，`nb_proba`、`svm_lin_proba`、`svm_rbf_proba`、`mlp_proba` 是正类概率。`source_row_index` 是原 CSV 经 pandas 读取后的从零开始的行索引，可以用于检查不同模型结果的样本顺序。

RBF 加权对照应保持本次 SVC 的其他参数，仅将 `class_weight` 改为 `'balanced'`。交叉验证按每折训练数据拟合预处理步骤。最终共同 Notebook 的集成由组内协作完成。

## 文件清单与验证

- `src/algorithm_engineering/run_models.py` 包含训练、指标、诊断和结果保存逻辑。
- `outputs/b_models/metrics.csv` 保存四项指标。
- `outputs/b_models/test_predictions.csv` 保存同一测试集的标签、预测类别和正类概率。
- `outputs/b_models/experiment_details.json` 保存参数、支持向量数、混淆矩阵、训练诊断、软件版本、源文件哈希和划分索引。
- `outputs/b_models/mlp_loss.csv` 与 `fig04_mlp_loss.png` 保存 MLP 训练过程。
- `reports/B_AI_USAGE.md` 记录 B 阶段 AI 使用情况。

本次已使用真实 CSV 完成四模型训练。指标和混淆矩阵由保存的逐样本预测再次计算核对，MLP 曲线已目视检查。A 的既有测试继续通过。
