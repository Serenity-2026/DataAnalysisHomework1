# A 数据工程师 EDA 摘要

## 数据与质量检查

- 原始数据：6014 行、21 列。
- 普通 `isna()` 检出的缺失：最后一条截断记录除 `customerID` 和残缺的 `gender` 外有 19 个字段缺失；缺失标签行共 1 条。
- 全字段均为空的记录：0 条；严重不完整记录：1 条。
- 完全重复行：0 条；重复 `customerID`：0 条。
- `TotalCharges` 原始类型：`object`；纯空白字符串 9 条；非空文本转换失败 0 条。
- `TotalCharges` 数值转换后缺失 10 条，其中 tenure=0 为 9 条，另有 1 条来自截断记录。

## 标签分布

剔除缺少标签的截断记录后，描述性分析使用 6013 条客户记录。
`No` 为 4424 条（73.57%），`Yes` 为 1589 条（26.43%）。
标签存在明显数量差异，后续模型评估不能只看准确率，还应结合灵敏度、特异度和 AUC；本模块不训练或比较模型。

## 特征观察

- `Contract` 中流失率最高的观测组为 `Month-to-month`（42.39%，n=3329）。
- `InternetService` 中流失率最高的观测组为 `Fiber optic`（41.84%，n=2641）。
- tenure 与 MonthlyCharges 的箱线图按真实 Churn 标签展示分布差异；这些差异是描述性关联，不能解释为因果关系。

## 相关性热力图

热力图排除 `customerID`，纳入数值特征、二元 Churn 标签，以及对类别特征执行 `drop_first=True` 后的独热编码列。该编码仅用于全量描述性相关分析，不用于模型训练/测试划分。
绝对相关系数最高的预测特征对为 `PhoneService_Yes` 与 `MultipleLines_No phone service`（r=-1.000）。
这说明 GaussianNB 的条件独立假设需要结合这些相关特征进一步讨论，但不能仅凭相关系数断言模型表现。

绝对相关系数较高的前五组预测特征：

- `PhoneService_Yes` / `MultipleLines_No phone service`：r=-1.000
- `OnlineBackup_No internet service` / `DeviceProtection_No internet service`：r=1.000
- `DeviceProtection_No internet service` / `StreamingTV_No internet service`：r=1.000
- `OnlineSecurity_No internet service` / `OnlineBackup_No internet service`：r=1.000
- `InternetService_No` / `StreamingMovies_No internet service`：r=1.000

## 图表文件

- `01_churn_distribution.png`
- `02_contract_churn_rate.png`
- `03_tenure_by_churn.png`
- `04_monthly_charges_by_churn.png`
- `05_internet_service_churn_rate.png`
- `06_feature_correlation_heatmap.png`
