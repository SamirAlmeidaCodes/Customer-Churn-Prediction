"""Paths, feature lists, and decision thresholds shared by training and serving."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
EXAMPLES_PATH = PROJECT_ROOT / "examples" / "customers.json"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"

# Canonical model matrix. ColumnTransformer selects by name, so this order is
# the API and form order rather than a numeric layout.
FEATURE_COLUMNS = [
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
]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    column for column in FEATURE_COLUMNS if column not in NUMERIC_FEATURES
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Outreach priority. Separate from the 0.5 cutoff that labels Churn vs Stay.
LOW_RISK_BELOW = 0.30
MEDIUM_RISK_BELOW = 0.60
CLASSIFICATION_THRESHOLD = 0.50

TENURE_MIN = 0
TENURE_MAX = 120
MONTHLY_CHARGES_MAX = 500.0
TOTAL_CHARGES_MAX = 20_000.0

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "subsample": 0.8,
    "random_state": RANDOM_STATE,
}
SMOTE_PARAMS = {
    "random_state": RANDOM_STATE,
    "k_neighbors": 5,
}

MODEL_NAMES = ("logistic_regression", "gradient_boosting")
