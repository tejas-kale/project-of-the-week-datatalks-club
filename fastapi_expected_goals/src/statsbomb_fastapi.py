import json
from typing import Union, List

from fastapi import FastAPI, Path, Request, HTTPException


def load_competitions() -> list:
    """
    Load `competitions.json`.
    :return: List of dictionaries, each containing information about a competition.
    """
    with open("../data/statsbomb/data/competitions.json", "r") as f:
        return json.load(f)


app = FastAPI()


@app.get("/")
def index(request: Request) -> dict:
    """
    Response on index page.
    :param request: FastAPI's Request object to get IP address of the host
    running the API server
    :return: Dictionary containing the key `message` and the value directing
    towards documentation.
    """
    return {"message": f"Visit {request.client.host}:8000/docs for the API documentation."}


@app.get("/list/competitions")
def list_competitions() -> dict:
    """
    List the name and ID of all competitions in `competitions.json`.
    :return: Dictionary of competition names and IDs.
    """
    competitions: list = load_competitions()

    response: dict = {}
    competition: dict
    for competition in competitions:
        competition_name: str = competition["competition_name"]
        response[competition_name] = {
            "name": competition_name,
            "id": competition["competition_id"]
        }

    return response


"""
Using the `Path` function, we get two benefits:
- A value of `None` prevents execution of the endpoint without any input in Swagger documentation.
- The description is provided above the text box for the input in Swagger documentation.
- Check for valid inputs. With `gt=0` below, we ask FastAPI to check if the input competition ID
is greater than 0. Other valid checks are `lt` (less than), `ge` (greater than or equal to), and
`le` (less than or equal to).
"""


@app.get("/competitions/id/{competition_id}")
def get_competition_info(
        competition_id: int = Path(0,
                                   description="ID of the competition whose information is to be fetched",
                                   gt=0)
) -> Union[List[dict], dict]:
    """
    Return all information of the competition specified by its ID. As we have data for multiple
    seasons, there can be more than one dictionary.
    :param competition_id: ID of the competition whose information is to be fetched.
    :return: List of dictionaries having information about the specified competitions.
    """
    competitions: list = load_competitions()

    competition: dict
    competition_info: list = []
    for competition in competitions:
        if competition["competition_id"] == competition_id:
            competition_info.append(competition)

    if len(competition_info) > 0:
        return competition_info
    else:
        raise HTTPException(status_code=404, detail="Invalid competition ID")


@app.get("/competitions/name")
def get_competition_info_by_name(n: str) -> Union[List[dict], dict]:
    """
    Get information about a competition by its name. As we have data for multiple
    seasons, there can be more than one dictionary.
    :param n: Name of the competition for which information is sought.
    :return: List of dictionaries providing information about one or more competitions that match
    the specified name.
    """
    competitions: list = load_competitions()

    competition: dict
    competition_info: list = []
    for competition in competitions:
        if competition["competition_name"] == n:
            competition_info.append(competition)

    if len(competition_info) > 0:
        return competition_info
    else:
        raise HTTPException(status_code=404, detail="Invalid competition name")
