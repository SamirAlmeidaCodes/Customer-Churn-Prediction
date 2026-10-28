"""Shared fixtures for API tests."""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from churn.config import EXAMPLES_PATH


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def examples():
    return json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
