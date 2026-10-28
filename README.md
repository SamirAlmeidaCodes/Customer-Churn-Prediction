# Customer churn prediction

Score whether a telco customer is likely to leave. The project trains a scikit-learn feature pipeline on the public IBM Telco Customer Churn dataset, compares logistic regression with gradient boosting, saves the stronger model with joblib, and serves it from FastAPI.

## Dataset

The file `data/Telco-Customer-Churn.csv` is the IBM Telco Customer Churn extract (7,043 customers, 21 columns), from the public [telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) repository. If that file is missing, `python -m churn.train` downloads it.

`Churn = Yes` is the positive class, about 26.5% of rows. `customerID` is an identifier and is not a feature. Eleven tenure-0 accounts have a blank `TotalCharges`. Those blanks are missing zeros, so cleaning fills them with 0. Any other numeric gap is median-imputed inside the model pipeline.

## Pipeline

Logistic regression and gradient boosting share this pipeline:

1. Numeric columns `tenure`, `MonthlyCharges`, and `TotalCharges`: median impute, then `StandardScaler`.
2. Categorical columns, including `SeniorCitizen`: most-frequent impute, then one-hot encoding. The encoding is dense, and unknown levels become zeros at score time.
3. SMOTE balances the training labels after encoding and scaling.
4. The classifier: `LogisticRegression` or `GradientBoostingClassifier` (200 trees, learning rate 0.05, max depth 3).

SMOTE is an `imblearn` sampler step on a scikit-learn preprocessor. The sampler runs during `fit` only, including inside each cross-validation fold. The holdout split and the API score the original class mix. SMOTE on one-hot columns can place a synthetic row between category codes. That is a consequence of running it after encoding, which is the order this project uses.

The winning pipeline is written to `models/churn_model.joblib`. Metrics, versions, and hyperparameters are written to `models/metrics.json`.

## Evaluation

The split is stratified 80/20 with `random_state=42`. Model choice uses 5-fold stratified cross-validation on the training split. The saved model is the one with the higher mean F1. Mean ROC-AUC breaks a tie. The holdout split is scored once and is not used to choose or refit the saved model.

Precision, recall, and F1 use the churn class at a probability cutoff of 0.50. ROC-AUC uses the churn probability.

### Results

Gradient boosting is the saved model. Selection uses mean cross-validated F1, which is 0.632 for gradient boosting and 0.630 for logistic regression. On the holdout split both models reach an F1 of about 0.617. Gradient boosting is the more precise of the two. Logistic regression recalls more churners. Holdout ROC-AUC is 0.843 for gradient boosting and 0.840 for logistic regression.

| Model | Split | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | Cross-validation | 0.526 | 0.788 | 0.630 | 0.845 |
| Logistic regression | Holdout | 0.503 | 0.797 | 0.617 | 0.840 |
| Gradient boosting | Cross-validation | 0.591 | 0.681 | 0.632 | 0.847 |
| Gradient boosting | Holdout | 0.564 | 0.679 | 0.617 | 0.843 |

Cross-validation cells are means across five training folds. `models/metrics.json` stores six decimal places and the fold standard deviations.

Risk tiers use the probability rounded to 4 decimal places:

| Tier | Probability |
| --- | --- |
| Low | under 0.30 |
| Medium | 0.30 up to, but not including, 0.60 |
| High | 0.60 and above |

The response field `prediction` is `Churn` when that probability is 0.50 or higher, and `Stay` otherwise. The tier is an outreach priority. It is separate from the Churn/Stay label.

## Project layout

```text
churn/                data cleaning, pipeline, training, scoring
app/                  FastAPI app, schema, and HTML form
data/                 Telco Customer Churn CSV
examples/             high-risk and low-risk request bodies
models/               joblib pipeline and metrics.json (created by training)
notebooks/            EDA and model comparison
tests/                pytest suite
Dockerfile
requirements.txt      runtime, including training
requirements-dev.txt  tests and the notebook
```

## Setup

From the project root, with Python 3.14:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m churn.train
pytest
uvicorn app.main:app --reload
```

If `python3 -m venv` complains that `ensurepip` is missing, create the environment with `python3 -m venv --without-pip .venv` and install pip into it before the requirements step.

Open http://127.0.0.1:8000 for the form. The form posts JSON to `/predict` and shows the probability and risk tier. Interactive docs are at http://127.0.0.1:8000/docs.

## API

`POST /predict` accepts one customer record. Field names match the training columns.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"gender":"Female","SeniorCitizen":1,"Partner":"No","Dependents":"No","tenure":1,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"Yes","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":95.5,"TotalCharges":95.5}'
```

A successful body looks like this:

```json
{"churn_probability": 0.9246, "risk_tier": "High", "prediction": "Churn"}
```

That body is the saved gradient boosting model scoring the short-tenure fiber profile in the curl command. The long-tenure two-year profile in `examples/customers.json` scores `0.0162`, tier Low, prediction Stay. The form can load either profile.

`GET /health` returns `{"status": "ok"}`.

The API checks each field against the categories present in the dataset. It does not reject inconsistent combinations, such as `PhoneService = No` together with `MultipleLines = Yes`.

## Tests

`pytest` covers cleaning, pipeline step order, SMOTE label balance, tier boundaries, the request schema, the saved metrics, and the HTML form plus the JSON endpoint. Train the model before the full suite so the artifact tests and the API tests have a pipeline to load.

## Notebook

`notebooks/eda_and_model_comparison.ipynb` walks through the class balance, charge and tenure distributions, churn rates by contract and service, the shared pipeline, and the holdout comparison. Executed outputs are saved in the file. To rerun it:

```bash
python -m ipykernel install --sys-prefix --name churn-pred
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=churn-pred \
  notebooks/eda_and_model_comparison.ipynb
```

Run `python -m churn.train` first if you want the notebook to check its holdout numbers against `models/metrics.json`.

## Docker

The image installs the runtime requirements, trains the model, and serves the API as a non-root user on port 8000.

```bash
docker build -t churn-prediction .
docker run --rm -p 8000:8000 churn-prediction
```

The build trains the model, so the first build is slower than the container start that follows. Then open http://127.0.0.1:8000.

## Reproducibility

`models/metrics.json` records the split, the SMOTE settings, both hyperparameter sets, the library versions, cross-validation means, and the holdout metrics. Training uses `random_state=42` for the split, the folds, SMOTE, and both classifiers.
