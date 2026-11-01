# Customer churn prediction

Score whether a telco customer is likely to leave. The project trains a scikit-learn feature pipeline on the public IBM Telco Customer Churn dataset, compares logistic regression with a tree model, saves the stronger model with joblib, and serves it from FastAPI. The training recipe depends on how many rows and how many churners the table has.

## After review

I personally reviewed the project, and training was changed so the recipe follows the size of the table. The same three tables were trained with the old fixed recipe and with the new one. The old recipe always used an 80/20 split, 5 folds, SMOTE with 5 neighbors, and 200 gradient-boosting trees. Each pair is one run on the same machine.

| Table | Time | Processor time | Peak memory |
| --- | --- | --- | --- |
| 800 rows (`data/demo_small.csv`) | 2.5s to 1.3s, 48% faster | 7.9s to 6.8s, 14% less | 191 MB to 190 MB, 0.3% less |
| 7,043 rows (`data/Telco-Customer-Churn.csv`) | 12.3s to 11.2s, 9% faster | 23.2s to 21.9s, 6% less | 207 MB to 208 MB, 0.3% more |
| 12,000 rows (`data/demo_large.csv`) | 23.4s to 2.7s, 89% faster | 93.4s to 20.4s, 78% less | 219 MB to 213 MB, 2% less |

The large table is where the clock time and the processor time drop. Histogram boosting replaces the 200-tree model there. Peak memory stays near the cost of loading the libraries. On the 7,043-row file the API serves, holdout precision moved from 0.564 to 0.567.

## Dataset

The file `data/Telco-Customer-Churn.csv` is the IBM Telco Customer Churn extract (7,043 customers, 21 columns), from the public [telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) repository. If that file is missing, `python -m churn.train` downloads it. This is the table the API serves. It has 1,869 churners, so training selects the medium regime.

Two demo tables sit beside it, both built from that extract by `python -m churn.demo_data`:

- `data/demo_small.csv` has 800 real customers (212 churners), sampled so the churn rate stays the same. Training selects the small regime.
- `data/demo_large.csv` has 12,000 rows (3,184 churners). Each row is a redrawn customer with tenure and charges nudged, and the churn label stays with the source row. Training selects the large regime.

`Churn = Yes` is the positive class, about 26.5% of rows in the source extract. `customerID` is an identifier and is not a feature. Eleven tenure-0 accounts in the source have a blank `TotalCharges`. Those blanks are missing zeros, so cleaning fills them with 0. Any other numeric gap is median-imputed inside the model pipeline.

Train a demo without replacing the served model:

```bash
python -m churn.train data/demo_small.csv
python -m churn.train data/demo_large.csv
```

Those commands write `models/demo_small_metrics.json` and `models/demo_large_metrics.json`. Only training the source extract writes `models/churn_model.joblib`.

## How table size changes the model

`churn/regimes.py` counts the rows and the churners, then picks one regime. Logistic regression is always the other candidate. Its solver stays `lbfgs` with at most 1,000 iterations. It uses balanced class weights only when the regime uses class weights.

| Regime | When | Split | Folds | Tree model | Class balance |
| --- | --- | --- | --- | --- | --- |
| Small | Under 1,000 rows, or under 150 churners | 70/30 under 1,000 rows, otherwise 80/20 | 5 | 100 trees, depth 2, learning rate 0.1, at least 5 rows in each leaf | SMOTE with 3 neighbors, or class weights when a training fold would hold fewer than about 30 churners |
| Medium | 1,000 to 10,000 rows, and at least 150 churners | 80/20 | 5 | 200 trees, depth 3, learning rate 0.05, 80% of rows per tree, early stopping | SMOTE with 5 neighbors |
| Large | Above 10,000 rows, and at least 150 churners | 90/10 when that holdout still has at least 1,000 rows and 200 churners, otherwise 80/20 | 3 | Histogram boosting, at most 300 trees, depth 3, learning rate 0.05, early stopping | Class weights once the training split has at least 2,000 churners, otherwise SMOTE with 5 neighbors |

The small regime spends a larger share of a short table on the holdout and fits a smaller tree model, so the run finishes sooner and is less able to memorize one customer. The medium regime is the original recipe with an early-stopping check on the 200 trees. On the 7,043-row file that check kept all 200. The large regime trains on a larger share of the rows, scores fewer folds, skips SMOTE when thousands of real churners are already present, and uses histogram boosting. That is the model behind the 89% shorter run on `data/demo_large.csv`.

A table above 10,000 rows with fewer than 150 churners still uses the small tree settings. A large table whose 10% holdout would contain fewer than 200 churners keeps the 80/20 split.

## Pipeline

Every regime shares the same preprocessing:

1. Numeric columns `tenure`, `MonthlyCharges`, and `TotalCharges`: median impute, then `StandardScaler`.
2. Categorical columns, including `SeniorCitizen`: most-frequent impute, then one-hot encoding. The encoding is dense, and unknown levels become zeros at score time.

Class balance and the tree model then follow the regime table. SMOTE, when it is used, is an `imblearn` sampler step. It runs during `fit` only, including inside each cross-validation fold. The holdout split and the API score the original class mix. SMOTE on one-hot columns can place a synthetic row between category codes. That is a consequence of running it after encoding, which is the order this project uses. When the regime uses class weights instead, no synthetic rows are added.

The winning pipeline for the source extract is written to `models/churn_model.joblib`. Metrics, the chosen regime, versions, and hyperparameters are written to `models/metrics.json`.

## Evaluation

Model choice uses stratified cross-validation on the training split. The saved model is the one with the higher mean F1. Mean ROC-AUC breaks a tie. The holdout split is scored once and is not used to choose or refit the saved model. The source extract uses an 80/20 split and 5 folds because it falls in the medium regime. The split is stratified with `random_state=42`.

Precision, recall, and F1 use the churn class at a probability cutoff of 0.50. ROC-AUC uses the churn probability.

### Results

These numbers are the medium regime on `data/Telco-Customer-Churn.csv`, which is the served model. Gradient boosting is saved. Its mean cross-validated F1 is 0.634, and logistic regression's is 0.630. On the holdout split, gradient boosting reaches an F1 of 0.619 and logistic regression reaches 0.617. Gradient boosting is the more precise of the two. Logistic regression recalls more churners. Holdout ROC-AUC is 0.841 for gradient boosting and 0.840 for logistic regression.

| Model | Split | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | Cross-validation | 0.526 | 0.788 | 0.630 | 0.845 |
| Logistic regression | Holdout | 0.503 | 0.797 | 0.617 | 0.840 |
| Gradient boosting | Cross-validation | 0.593 | 0.682 | 0.634 | 0.846 |
| Gradient boosting | Holdout | 0.567 | 0.682 | 0.619 | 0.841 |

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
churn/                data cleaning, regimes, pipeline, training, scoring
app/                  FastAPI app, schema, and HTML form
data/                 source CSV, demo_small.csv, demo_large.csv
examples/             high-risk and low-risk request bodies
models/               joblib pipeline, metrics.json, demo metrics
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
{"churn_probability": 0.9310, "risk_tier": "High", "prediction": "Churn"}
```

That body is the saved gradient boosting model scoring the short-tenure fiber profile in the curl command. The long-tenure two-year profile in `examples/customers.json` scores `0.0179`, tier Low, prediction Stay. The form can load either profile.

`GET /health` returns `{"status": "ok"}`.

The API checks each field against the categories present in the dataset. It does not reject inconsistent combinations, such as `PhoneService = No` together with `MultipleLines = Yes`.

## Tests

`pytest` covers cleaning, regime boundaries, the demo table sizes, pipeline step order, SMOTE label balance, tier boundaries, the request schema, the saved metrics, and the HTML form plus the JSON endpoint. Train the model before the full suite so the artifact tests and the API tests have a pipeline to load.

## Notebook

`notebooks/eda_and_model_comparison.ipynb` walks through the class balance, charge and tenure distributions, churn rates by contract and service, the shared pipeline, and the holdout comparison. The source extract falls in the medium regime, so a fresh execution uses that recipe. Executed outputs are saved in the file. To rerun it:

```bash
python -m ipykernel install --sys-prefix --name churn-pred
python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=churn-pred \
  notebooks/eda_and_model_comparison.ipynb
```

Run `python -m churn.train` first if you want the notebook to check its holdout numbers against `models/metrics.json`.

## Docker

The image installs the runtime requirements, trains the model on the source extract, and serves the API as a non-root user on port 8000.

```bash
docker build -t churn-prediction .
docker run --rm -p 8000:8000 churn-prediction
```

The build trains the model, so the first build is slower than the container start that follows. Then open http://127.0.0.1:8000.

## Reproducibility

`models/metrics.json` records the regime, the split, the SMOTE or class-weight settings, both hyperparameter sets, the library versions, cross-validation means, and the holdout metrics. Training uses `random_state=42` for the split, the folds, SMOTE, and both classifiers.
