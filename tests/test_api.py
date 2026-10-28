"""Tests for the form and the JSON scoring endpoint."""

from churn.config import FEATURE_COLUMNS
from churn.risk import churn_label, risk_tier


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_form_lists_every_feature_and_posts_to_predict(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert 'id="score-form"' in html
    assert "/predict" in html
    assert "churn_probability" in html
    assert 'data-example="high_risk"' in html
    assert 'data-example="low_risk"' in html
    for column in FEATURE_COLUMNS:
        assert f'name="{column}"' in html


def test_predict_returns_probability_and_tier(client, examples):
    response = client.post("/predict", json=examples["high_risk"])
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"churn_probability", "risk_tier", "prediction"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_probability"] == round(body["churn_probability"], 4)
    assert body["risk_tier"] == risk_tier(body["churn_probability"])
    assert body["prediction"] == churn_label(body["churn_probability"])


def test_short_tenure_fiber_scores_above_the_long_contract(client, examples):
    high = client.post("/predict", json=examples["high_risk"]).json()
    low = client.post("/predict", json=examples["low_risk"]).json()
    assert high["churn_probability"] > low["churn_probability"]


def test_invalid_record_returns_422(client, examples):
    payload = dict(examples["high_risk"])
    payload["gender"] = "Other"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    missing = dict(examples["high_risk"])
    del missing["Contract"]
    assert client.post("/predict", json=missing).status_code == 422

    broken = client.post(
        "/predict",
        content="{",
        headers={"content-type": "application/json"},
    )
    assert broken.status_code == 422


def test_openapi_documents_predict(client):
    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    paths = spec.json()["paths"]
    assert "post" in paths["/predict"]
    assert "200" in paths["/predict"]["post"]["responses"]
    assert "422" in paths["/predict"]["post"]["responses"]
