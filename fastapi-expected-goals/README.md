# Building an expected goals API using FastAPI

Starting December 7, 2022, Datatalks.Club's `#project-of-the-week` involves 
building an API using FastAPI to serve a machine learning model. As part of 
the project, I will build a logistic regression model for computing expected
goals from shots in a football match. The expected API is expected to be 
simple - given the (X, Y) coordinates of a shot and the type of shot (regular,
header, or free kick), the API will return an expected goal value which
is the probability of the shot resulting in a goal.

The technical details for building this logistic regression model were 
learnt from the excellent 
[Mathematical Modelling of Football](https://uppsala.instructure.com/courses/28112/pages/2-statistical-models-of-actions)
course put together by the club analysts who also created the 
[Friends of Tracking](https://www.youtube.com/channel/UCUBFJYcag8j2rm_9HkrrA7w)
YouTube channel. The data for the model comes from the data provider Wyscout 
and can be downloaded [here](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5).
Additional data, provided by StatsBomb, can be found
[here](https://github.com/statsbomb/open-data).