import json

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from fastapi_expected_goals.src.xg_endpoint import app

client: TestClient = TestClient(app=app)


def test_compute_xg_returns_correct_response():
    """
    Test that the correct xG value is returned for the specified coordinates.
    """
    shot_coordinates: dict = {
        "xc": 91.6,
        "yc": 69.3
    }
    expected_xg_value: float = 0.091168

    response: Response = client.post("/predict", json=shot_coordinates)
    content: dict = json.loads(response.content)
    assert content["shot_xg"] == pytest.approx(expected_xg_value, 0.01)


def test_compute_xg_returns_error_response():
    """
    Test that an error is given when invalid shot coordinates are provided.
    """
    shot_coordinates: dict = {
        "xc": -20.0,
        "yc": 79.3
    }

    response: Response = client.post("/predict", data=shot_coordinates)
    assert response.status_code == 422
