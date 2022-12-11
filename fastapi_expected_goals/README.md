# Building an expected goals API using FastAPI

## Usage

This endpoint predict the xG value of a shot given its location on the pitch. To test this endpoint, navigate to the `src` directory and start an API server by running the following command in the terminal:

```commandline
uvicorn xg_endpoint:app --reload
```

Once running, navigate to `127.0.0.1:8000/docs` in a browser, click on */predict* in the Green box, and then click on *Try it out*. In the *Request body* box that opens up, enter the X (`xc`) and Y (`yc`) coordinates of shot (in Wyscout units). An example shot location is:

```python
{
    "xc": 91.6,
    "yc": 69.3
}
```

Scroll down and click on *Execute*. Scroll down further and in the *Server response* section, the following response body should be seen:

```python
{
  "shot_xg": 0.09116846274823091
}
```

## Background

Starting December 7, 2022, Datatalks.Club's `#project-of-the-week` involves building an API using FastAPI to serve a machine learning model. As part of the project, I will build a logistic regression model for computing expected goals (xG) from shots in a football match. The expected API is expected to be simple - given the (X, Y) coordinates of a shot and the type of shot (regular, header, or free kick), the API will return an expected goal value which is the probability of the shot resulting in a goal.

The technical details for building this logistic regression model were learnt from the excellent [Mathematical Modelling of Football](https://uppsala.instructure.com/courses/28112/pages/2-statistical-models-of-actions) course put together by the club analysts who also created the [Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)
YouTube channel. The data for the model comes from the data provider Wyscout and can be downloaded [here](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5). Additional data, provided by StatsBomb, can be found [here](https://github.com/statsbomb/open-data).

Exploratory analysis of the Statsbomb data can be seen in [nbs/00_leading_investigating_world_cup_data.ipynb](nbs/00_loading_investigating_world_cup_data.ipynb). It talks about how the data is structured in different files, how to read event data for a specific match, and plots shots and passes from the event data. [nbs/01_building_xg_model.ipynb](nbs/01_building_xg_model.ipynb) builds a xG model using events from La Liga (Spanish league) and saves the model (i.e. Logistic Regression parameters) to a `.pkl` file.

[src/hello_fastapi.py](./src/hello_fastapi.py) is my introduction to FastAPI. With [src/statsbomb_fastapi.py](./src/statsbomb_fastapi.py), I attempt to write GET methods that use either dynamic paths or query parameters to fetch information about football competitions covered in Statsbomb data. With [tests/test_statsbomb_fastapi.py](./tests/test_statsbomb_api.py), I add unit tests for these GET methods.