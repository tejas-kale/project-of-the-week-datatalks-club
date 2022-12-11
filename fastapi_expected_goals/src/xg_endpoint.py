import os
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

MODEL_DIR: str = "../models"
MODEL_FN: str = "baseline_logistic_model.pkl"


class ShotPosition(BaseModel):
    xc: float = Field(description="X coordinate of the shot taken (in Wyscout units).")
    yc: float = Field(description="Y coordinate of the shot taken (in Wyscout units).")


app = FastAPI()


@app.get("/")
def index(request: Request):
    """
    Response on index page.
    :param request: FastAPI's Request object to get IP address of the host
    running the API server
    :return: Dictionary containing the key `message` and the value directing
    towards documentation.
    """
    return {"message": f"Visit {request.client.host}:8000/docs for the API documentation."}


def transform_coordinates_for_computation(x: float, y: float) -> (float, float):
    """
    Question: What does this transformation do?
    :param x: X coordinate of the shot taken.
    :param y: Y coordinate of the shot taken
    :return: Tuple of transformed X, Y coordinates.
    """
    m_x: float = 100 - x
    c: float = abs(y - 50)

    return (m_x * 105) / 100, (c * 65) / 100


def compute_shot_distance(x: float, y: float) -> float:
    """
    Computes distance of a shot from goal.
    :param x: X coordinate of the shot taken.
    :param y: Y coordinate of the shot taken.
    :return: Float denoting distance (in meters) of the shot from goal.
    """
    t_x: float
    t_y: float
    t_x, t_y = transform_coordinates_for_computation(x, y)

    return np.sqrt(t_x ** 2 + t_y ** 2)


def compute_shot_angle(x: float, y: float) -> float:
    """
    Computes the angle of the shot to the width of the goal post.
    :param x: X coordinate of the shot taken.
    :param y: Y coordinate of the shot taken
    :return: Float denoting the angle (in radians) of the shot.
    """
    t_x: float
    t_y: float
    t_x, t_y = transform_coordinates_for_computation(x, y)

    angle: float = np.arctan((7.32 * t_x) / (t_x ** 2 + t_y ** 2 - (7.32 / 2) ** 2))
    if angle < 0:
        angle = np.pi + angle

    return angle


"""
While it seems odd to create a POST request for fetching predictions from our model,
as the predictions depend on inputs that need to be sent as API request body, it is
a better choice than a GET request.
"""


@app.post("/predict",
          summary="Compute xG",
          description="Compute xG from shot location."
          )
def compute_xg(shot_position: ShotPosition) -> dict:
    """
    Compute the xG of a shot given the (X, Y) coordinates of where it was taken from.
    :param shot_position: Object of ShotPosition class that specifies the X- and Y-coordinate
    of where the shot was taken from (in Wyscout units).
    :return: A dictionary providing xG value.
    """
    xc: float = shot_position.xc
    yc: float = shot_position.yc

    shot_dist: float = compute_shot_distance(xc, yc)
    shot_ang: float = compute_shot_angle(xc, yc)

    with open(os.path.join(MODEL_DIR, MODEL_FN), "rb") as pf:
        model_params: pd.Series = pickle.load(pf)

    linear_sum: float = (model_params["Intercept"] +
                         (shot_dist * model_params["dist"]) +
                         (shot_ang * model_params["angle"]))

    return {"shot_xg": 1 / (1 + np.exp(-1 * linear_sum))}
