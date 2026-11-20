# A 数据工程师任务一与任务二交付说明

## 1. 工作范围

本交付只覆盖任务一探索性数据分析和任务二数据预处理。A 不训练或比较 GaussianNB、SVM、MLP、决策树、随机森林，也不生成最终实验报告。

## 2. 真实数据核验

`data/telco_churn.csv` 实际为 6014 行、21 列，字段顺序与任务书一致。原始文件 SHA-256 为 `E96D3A2C6731334D0EC12197C564870D74B988477FA842AEFB847377785D4F15`。

数据质量检查得到：

- `TotalCharges` 原始类型为 `object`，有 9 个纯空白字符串；非空文本转数值失败为 0。
- 这 9 个空白累计费用全部对应 `tenure == 0`。
- 文件最后一行仅有 `customerID=5671-UUNXD` 和残缺的 `gender=Fem`，其余 19 个字段（包括 `Churn`）缺失，因此不能用于监督学习。
- 完全重复行、缺失或重复 `customerID` 均为 0。
- 删除缺少标签的截断行后，有效记录 6013 条；`No` 4424 条（73.57%），`Yes` 1589 条（26.43%）。

完整机器可读计数见 `outputs/eda/data_quality_summary.json`，描述性结论见 `outputs/eda/eda_summary.md`。

## 3. 任务一 EDA 产出

`python -m src.data_engineering.eda` 生成：

1. `01_churn_distribution.png`：标签计数与比例。
2. `02_contract_churn_rate.png`：不同合同类型的流失率与样本数。
3. `03_tenure_by_churn.png`：按标签分组的 tenure 箱线图。
4. `04_monthly_charges_by_churn.png`：按标签分组的月费箱线图。
5. `05_internet_service_churn_rate.png`：不同网络服务的流失率与样本数。
6. `06_feature_correlation_heatmap.png`：数值特征与 `drop_first=True` 类别编码特征的相关性热力图。

热力图排除 `customerID`。绝对相关系数最高的预测特征对是 `PhoneService_Yes` 与 `MultipleLines_No phone service`（r=-1.000）；多个“No internet service”哑变量也呈完全相关。这些结果说明 GaussianNB 的条件独立假设需要结合冗余服务状态特征讨论，但不能仅凭相关系数断言模型表现。

## 4. 任务二清洗规则

清洗顺序如下：

1. 复制输入 DataFrame，原始 CSV 不被修改。
2. 去除字符串首尾空格，把空字符串转换为缺失值。
3. 删除全空行；对缺少 `Churn` 的行单独计数并删除。本数据实际没有全字段均空行，但有 1 条严重不完整且缺少标签的截断行。
4. 完全重复行只在逐字段一致时删除并计数；不同内容的重复 `customerID` 不自动删除，而是明确报错要求人工核对。
5. `TotalCharges` 转换为数值。`tenure == 0` 的缺失值按尚未产生累计费用填 0；其他数值缺失仅使用训练集的中位数填补。
6. 其他类别缺失仅使用训练集众数填补。
7. `Churn` 映射为 `No=0`、`Yes=1`，未知标签明确报错。
8. `customerID` 在划分前从模型特征中删除。

## 5. 划分、编码与缩放

- 唯一一次划分使用 `test_size=0.2`、`random_state=42`、`stratify=y`。
- 划分后训练集 4810 条，测试集 1203 条。
- 类别水平只从训练集确定，随后使用 `pd.get_dummies(..., drop_first=True)`；测试集按训练列重排。
- 编码后共有 30 个特征，列表见 `outputs/preprocessing/feature_names.txt`。
- 数值中位数、类别众数和 `StandardScaler` 均只在训练集上拟合。
- `scaler.n_samples_seen_` 为 4810，与训练集样本数一致；测试集只调用同一 scaler 的 `transform`。
- 未缩放与缩放后的四个特征表均无 NaN 或无穷值。

## 6. 接口

```python
from src.data_engineering import prepare_data

prepared = prepare_data()
```

| 字段 | 内容 |
|---|---|
| `X_train` | 编码后、未缩放的训练特征 |
| `X_test` | 编码后、未缩放的测试特征 |
| `X_train_scaled` | 仅用训练集拟合 scaler 后的训练特征 |
| `X_test_scaled` | 使用同一 scaler 转换的测试特征 |
| `y_train` | 训练标签 |
| `y_test` | 测试标签 |
| `feature_names` | 固定特征列名和顺序 |
| `scaler` | 只在训练集拟合的 `StandardScaler` |
| `metadata` | 清洗、划分、编码、填补和缩放的追溯信息 |

GaussianNB 和树模型使用未缩放数据；SVM、MLP 使用缩放数据。5 折交叉验证必须使用未缩放的 `X_train`，并在每折 `Pipeline` 内拟合缩放器。

## 7. 验证

实际执行：

```text
python -m unittest discover -s tests -v
Ran 9 tests ... OK

python -m src.data_engineering.eda --force
python -m src.data_engineering.preprocessing --force
训练=(4810, 30)，测试=(1203, 30)，scaler.n_samples_seen_=4810
```

测试覆盖字段验证、隐藏空白、全空与截断行、`TotalCharges` 规则、训练集统计量填补、未知标签、重复 ID、列对齐、分层划分和缩放无泄漏。6 张 EDA 图均已目视检查。

## 8. 提交形式风险

任务书原始提交物包含可运行 Notebook。本项目按用户要求不创建或维护 Notebook，最终提交前需确认教师是否接受 Python 脚本替代。
