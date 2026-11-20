# 电信公司客户流失预警 A 数据工程

本仓库当前只实现 A 数据工程师负责的任务一探索性数据分析和任务二数据预处理。代码读取本地真实 CSV，生成可追溯的质量统计、EDA 图表及一套无数据泄漏的统一数据接口；不包含 GaussianNB、SVM、MLP、决策树、随机森林、模型评估或最终实验报告。

## 已验证的数据情况

当前本地 `data/telco_churn.csv` 实际读取为 6014 行、21 列，与任务书描述一致。真实运行发现：

- 最后一行仅含 `customerID=5671-UUNXD` 和残缺的 `gender=Fem`，其余 19 个字段缺失；该行缺少监督学习标签，预处理时删除并记录。
- `TotalCharges` 以文本类型读取，其中 9 条为纯空白字符串；这 9 条均对应 `tenure == 0`，按“尚未产生累计费用”的业务语义填为 0。
- 完全重复行、重复 `customerID` 和非空 `TotalCharges` 转换失败均为 0。
- 有效标签记录共 6013 条：`No` 4424 条（73.57%），`Yes` 1589 条（26.43%）。
- 固定分层划分后，训练集 4810 条、测试集 1203 条；独热编码后 30 个特征。

这些数字来自仓库内脚本的实际运行结果，详情见 `outputs/eda/` 和 `outputs/preprocessing/`。

## 项目结构

```text
.
├── data/README.md
├── src/data_engineering/
│   ├── config.py
│   ├── quality.py
│   ├── eda.py
│   └── preprocessing.py
├── tests/test_data_engineering.py
├── outputs/eda/
├── outputs/preprocessing/
├── reports/AI_USAGE.md
├── A数据工程师内容.md
└── TODO.md
```

原始 CSV 不纳入 Git。请将课程数据放到 `data/telco_churn.csv`。

## 环境与运行

建议在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.data_engineering.eda
python -m src.data_engineering.preprocessing
```

脚本默认拒绝覆盖已有同名产出。确认要重新生成时使用：

```powershell
python -m src.data_engineering.eda --force
python -m src.data_engineering.preprocessing --force
```

## 统一数据接口

```python
from src.data_engineering import prepare_data

data = prepare_data()

data.X_train
data.X_test
data.X_train_scaled
data.X_test_scaled
data.y_train
data.y_test
data.feature_names
data.scaler
```

接口固定使用 `test_size=0.2`、`random_state=42` 和 `stratify=y`。类别特征在训练集上确定类别顺序后使用 `pd.get_dummies(..., drop_first=True)`；数值中位数、类别众数和 `StandardScaler` 都只在训练集上拟合，测试集只应用相同规则。

- GaussianNB、基线、决策树和随机森林使用 `X_train`、`X_test`。
- SVM、MLP 使用 `X_train_scaled`、`X_test_scaled`。
- 交叉验证应从未缩放的 `X_train` 开始，并在每一折的 `Pipeline` 内拟合缩放器，不能对全部训练数据预先缩放。
- 后续成员应复用 `y_train`、`y_test` 和 `feature_names`，不要重新随机划分后直接比较模型。

更完整的清洗依据、接口字段和验收结果见 `A数据工程师内容.md`。

## 已执行验证

当前环境使用 Python 3.13.9，实际执行并通过：

```text
python -m unittest discover -s tests -v
Ran 9 tests ... OK
```

随后实际运行 EDA 与预处理命令，生成 6 张 EDA 图、质量摘要、特征清单和运行日志；所有图表均已目视检查。

## 提交形式提醒

任务书原始提交要求包含可运行 Notebook。本项目按用户要求仅使用 Python 模块、命令行、图表和 Markdown。最终提交前必须确认教师是否接受脚本替代 Notebook，不能把当前形式描述为完全符合原始提交格式。
