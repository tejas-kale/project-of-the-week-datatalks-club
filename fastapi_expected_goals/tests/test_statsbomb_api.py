import json

from fastapi.testclient import TestClient
from httpx import Response

from fastapi_expected_goals.src.statsbomb_fastapi import app

client: TestClient = TestClient(app=app)


def test_index_returns_correct_response():
    """
    Test that the index page shows the URL to the Swagger documentation.
    :return:
    """
    response: Response = client.get("/")
    content: dict = json.loads(response.content)
    assert content["message"] == "Visit testclient:8000/docs for the API documentation."


def test_get_competition_info_returns_correct_response():
    expected_competition_info: list = [
        {
            "competition_id": 37,
            "season_id": 90,
            "country_name": "England",
            "competition_name": "FA Women's Super League",
            "competition_gender": "female",
            "competition_youth": False,
            "competition_international": False,
            "season_name": "2020/2021",
            "match_updated": "2022-08-16T02:10:37.220648",
            "match_updated_360": "2021-06-13T16:17:31.694",
            "match_available_360": None,
            "match_available": "2022-08-16T02:10:37.220648"
        },
        {
            "competition_id": 37,
            "season_id": 42,
            "country_name": "England",
            "competition_name": "FA Women's Super League",
            "competition_gender": "female",
            "competition_youth": False,
            "competition_international": False,
            "season_name": "2019/2020",
            "match_updated": "2021-06-01T13:01:18.188",
            "match_updated_360": "2021-06-13T16:17:31.694",
            "match_available_360": None,
            "match_available": "2021-06-01T13:01:18.188"
        },
        {
            "competition_id": 37,
            "season_id": 4,
            "country_name": "England",
            "competition_name": "FA Women's Super League",
            "competition_gender": "female",
            "competition_youth": False,
            "competition_international": False,
            "season_name": "2018/2019",
            "match_updated": "2022-09-12T21:06:25.061309",
            "match_updated_360": "2021-06-13T16:17:31.694",
            "match_available_360": None,
            "match_available": "2022-09-12T21:06:25.061309"
        }
    ]

    response: Response = client.get("/competitions/id/37")
    content: dict = json.loads(response.content)
    assert content == expected_competition_info


def test_get_competition_info_returns_error_response():
    response: Response = client.get("/competitions/id/20")
    content: dict = json.loads(response.content)
    assert response.status_code == 404
    assert content["detail"] == "Invalid competition ID"

    response: Response = client.get("/competitions/id/-1")
    assert response.status_code == 422


def test_get_competition_info_by_name_returns_correct_response():
    expected_competition_info: list = [
        {
            "competition_id": 49,
            "season_id": 3,
            "country_name": "United States of America",
            "competition_name": "NWSL",
            "competition_gender": "female",
            "competition_youth": False,
            "competition_international": False,
            "season_name": "2018",
            "match_updated": "2021-11-06T05:53:29.435016",
            "match_updated_360": "2021-06-13T16:17:31.694",
            "match_available_360": None,
            "match_available": "2021-11-06T05:53:29.435016"
        }
    ]

    response: Response = client.get("/competitions/name?n=NWSL")
    content: dict = json.loads(response.content)
    assert content == expected_competition_info


def test_get_competition_info_by_name_returns_error_response():
    response: Response = client.get("/competitions/name?n=Europa League")
    content: dict = json.loads(response.content)
    assert response.status_code == 404
    assert content["detail"] == "Invalid competition name"
