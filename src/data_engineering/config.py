"""A 数据工程师任务使用的集中配置。"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"
PREPROCESSING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing"

TEST_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
