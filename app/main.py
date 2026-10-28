"""Score telco customers and serve a small form for manual checks."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.schemas import FORM_FIELDS, CustomerRecord, PredictionResponse
from churn import __version__
from churn.config import EXAMPLES_PATH, METRICS_PATH, PROJECT_ROOT
from churn.predict import load_model, predict_customer

TEMPLATES = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))


def load_examples(path=EXAMPLES_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(path=METRICS_PATH) -> dict:
    metrics_path = path
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found at {metrics_path}. Run `python -m churn.train` first."
        )
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def served_summary(metrics: dict) -> dict:
    name = metrics["best_model"]
    holdout = metrics["models"][name]["holdout"]
    return {
        "label": name.replace("_", " "),
        "precision": holdout["precision"],
        "recall": holdout["recall"],
        "f1": holdout["f1"],
        "roc_auc": holdout["roc_auc"],
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.examples = load_examples()
    app.state.served = served_summary(load_metrics())
    yield


app = FastAPI(
    title="Customer Churn Prediction",
    version=__version__,
    summary="Score a telco customer for churn risk.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    examples = request.app.state.examples
    examples_json = json.dumps(examples).replace("<", "\\u003c")
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "fields": FORM_FIELDS,
            "examples_json": examples_json,
            "default": examples["high_risk"],
            "served": request.app.state.served,
        },
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord, request: Request) -> dict:
    """Return the churn probability and risk tier for one customer."""
    return predict_customer(request.app.state.model, record.model_dump(mode="json"))
