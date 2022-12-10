# Loading and Investigating World Cup Data

In this notebook, we will understand how to load and inspect event data of Women's World Cup matches. We follow the Prof. David Sumpter's [video](https://www.youtube.com/watch?v=GTtuOt03FM0&ab_channel=FriendsofTracking) for understanding how to download the data and inspect it using Python. During the course of this notebook, we will assume that both Statsbomb and Wyscout data is available in the `data` directory. URLs to download the data are provided in the *References* section.

The event data is provided in JSON files, so we need to import the `json` package to load these files. We will need `matplotlib` to plot the data and `numpy` to transform the data.


```python
import json
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_utils import create_pitch
```

## Load data

First, we will use the Statsbomb data. Let us load information about the competitions for which data is available.


```python
with open("./data/statsbomb/data/competitions.json", "r") as f:
    competitions: list = json.load(f)
```

We have a list of 19 competitions covered in the Statsbomb data. Let us look at the information of the first competition.


```python
competitions[0]
```




    {'competition_id': 16,
     'season_id': 4,
     'country_name': 'Europe',
     'competition_name': 'Champions League',
     'competition_gender': 'male',
     'competition_youth': False,
     'competition_international': False,
     'season_name': '2018/2019',
     'match_updated': '2022-08-14T16:57:15.866765',
     'match_updated_360': '2021-06-13T16:17:31.694',
     'match_available_360': None,
     'match_available': '2022-08-14T16:57:15.866765'}



In this notebook, we want to inspect data for the 2019 Women's World Cup. Its competition ID is `72`.


```python
competition_id: int = 72
```

Let us load information about all matches from the competition.


```python
with open(f"./data/statsbomb/data/matches/{competition_id}/30.json", "r") as f:
    matches: list = json.load(f)
```

There were 52 matches played during the World Cup.


```python
len(matches)
```




    52



Let us now print the result of every match in the World Cup. It will help us understand the structure of match result.

While it would be better for readability to get `match["home_team"]["country"]["name"]`, the event data that we want to analyse specifies `match["home_team"]["home_team_name"]` for every event. The same applies for the away team as well.


```python
match: dict
for match in matches:
    home_team_name: str = match["home_team"]["home_team_name"]
    away_team_name: str = match["away_team"]["away_team_name"]
    home_score: int = match["home_score"]
    away_score: int = match["away_score"]
    print(f"The match between {home_team_name} and {away_team_name} finished {home_score}-{away_score}")
```

    The match between Jamaica Women's and Italy Women's finished 0-5
    The match between Jamaica Women's and Australia Women's finished 1-4
    The match between Norway Women's and Australia Women's finished 1-1
    The match between Australia Women's and Italy Women's finished 1-2
    The match between Argentina Women's and Japan Women's finished 0-0
    The match between United States Women's and Thailand Women's finished 13-0
    The match between Chile Women's and Sweden Women's finished 0-2
    The match between Nigeria Women's and Korea Republic Women's finished 2-0
    The match between Germany Women's and Spain Women's finished 1-0
    The match between South Africa Women's and China PR Women's finished 0-1
    The match between Japan Women's and Scotland Women's finished 2-1
    The match between Netherlands Women's and Cameroon Women's finished 3-1
    The match between Canada Women's and New Zealand Women's finished 2-0
    The match between Netherlands Women's and Canada Women's finished 2-1
    The match between Sweden Women's and United States Women's finished 0-2
    The match between Sweden Women's and Canada Women's finished 1-0
    The match between Cameroon Women's and New Zealand Women's finished 2-1
    The match between Germany Women's and Nigeria Women's finished 3-0
    The match between England Women's and Sweden Women's finished 1-2
    The match between France Women's and Brazil Women's finished 2-1
    The match between Germany Women's and Sweden Women's finished 1-2
    The match between England Women's and Scotland Women's finished 2-1
    The match between Sweden Women's and Thailand Women's finished 5-1
    The match between Japan Women's and England Women's finished 0-2
    The match between Korea Republic Women's and Norway Women's finished 1-2
    The match between United States Women's and Chile Women's finished 3-0
    The match between Norway Women's and Nigeria Women's finished 3-0
    The match between United States Women's and Netherlands Women's finished 2-0
    The match between China PR Women's and Spain Women's finished 0-0
    The match between Italy Women's and Brazil Women's finished 0-1
    The match between France Women's and Korea Republic Women's finished 4-0
    The match between Germany Women's and China PR Women's finished 1-0
    The match between Spain Women's and South Africa Women's finished 3-1
    The match between Brazil Women's and Jamaica Women's finished 3-0
    The match between Canada Women's and Cameroon Women's finished 1-0
    The match between France Women's and United States Women's finished 1-2
    The match between New Zealand Women's and Netherlands Women's finished 0-1
    The match between France Women's and Norway Women's finished 2-1
    The match between Australia Women's and Brazil Women's finished 3-2
    The match between England Women's and Argentina Women's finished 1-0
    The match between Nigeria Women's and France Women's finished 0-1
    The match between South Africa Women's and Germany Women's finished 0-4
    The match between Scotland Women's and Argentina Women's finished 3-3
    The match between Thailand Women's and Chile Women's finished 0-2
    The match between England Women's and Cameroon Women's finished 3-0
    The match between Italy Women's and China PR Women's finished 2-0
    The match between Netherlands Women's and Japan Women's finished 2-1
    The match between Norway Women's and England Women's finished 0-3
    The match between England Women's and United States Women's finished 1-2
    The match between Netherlands Women's and Sweden Women's finished 1-0
    The match between Italy Women's and Netherlands Women's finished 0-2
    The match between Spain Women's and United States Women's finished 1-2


Let us consider the final of the World Cup between the USA and Netherlands and find its match ID.


```python
required_home_team: str = "United States Women's"
required_away_team: str = "Netherlands Women's"
```


```python
required_match_id: Union[int, str] = "Not found"
for match in matches:
    home_team_name: str = match["home_team"]["home_team_name"]
    away_team_name: str = match["away_team"]["away_team_name"]
    if (home_team_name == required_home_team) and (away_team_name == required_away_team):
        required_match_id: int = match["match_id"]

print(f"{required_home_team} vs {required_away_team} has ID: {required_match_id}")
```

    United States Women's vs Netherlands Women's has ID: 69321


Let us now load the event data for this match based on its ID.


```python
with open(f"./data/statsbomb/data/events/{required_match_id}.json", "r") as f:
    match_events: list = json.load(f)
```

This is the event data that we can use for various purposes like creating different kinds of plot and building models like expected goals. The first part of this data contains information about lineups and formations. After that, all information about events that happened on the ball are captured. It includes passes, interceptions, shots, and other on-ball events. For a pass, the start and end coordinate (X, Y) are noted. For a shot, the (X, Y) coordinate from where the shot is taken is recorded as well as where the shot landed up (inside or outside the frame of the goal).

Let us transform this data into a Pandas dataframe so that it is easier to inspect.


```python
events: pd.DataFrame = (pd.json_normalize(match_events, sep="_")
                        .assign(match_id=required_match_id))
events.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>index</th>
      <th>period</th>
      <th>timestamp</th>
      <th>minute</th>
      <th>second</th>
      <th>possession</th>
      <th>duration</th>
      <th>type_id</th>
      <th>type_name</th>
      <th>...</th>
      <th>shot_aerial_won</th>
      <th>foul_committed_penalty</th>
      <th>foul_won_penalty</th>
      <th>50_50_outcome_id</th>
      <th>50_50_outcome_name</th>
      <th>pass_goal_assist</th>
      <th>pass_through_ball</th>
      <th>shot_one_on_one</th>
      <th>foul_committed_offensive</th>
      <th>match_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8f16692f-c15c-4664-9ca8-7ee41df124d5</td>
      <td>1</td>
      <td>1</td>
      <td>00:00:00.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0000</td>
      <td>35</td>
      <td>Starting XI</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>1</th>
      <td>27e4a371-5f0b-4c55-a63f-9a3b7c1decdb</td>
      <td>2</td>
      <td>1</td>
      <td>00:00:00.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0000</td>
      <td>35</td>
      <td>Starting XI</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bd75bde1-9e27-436b-bbbf-d1d8d92b17d1</td>
      <td>3</td>
      <td>1</td>
      <td>00:00:00.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0000</td>
      <td>18</td>
      <td>Half Start</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3aa460ba-0b79-40df-836c-80f2cec2ae3b</td>
      <td>4</td>
      <td>1</td>
      <td>00:00:00.000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0.0000</td>
      <td>18</td>
      <td>Half Start</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>4</th>
      <td>cda3adf5-db44-4d52-a722-8484b56e9418</td>
      <td>5</td>
      <td>1</td>
      <td>00:00:00.140</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1.2136</td>
      <td>30</td>
      <td>Pass</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 117 columns</p>
</div>



This is a large dataframe with 117 columns! Let us filter it to only include data about shots.


```python
shots: pd.DataFrame = events.loc[events["type_name"] == "Shot"].set_index("id")
shots.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>period</th>
      <th>timestamp</th>
      <th>minute</th>
      <th>second</th>
      <th>possession</th>
      <th>duration</th>
      <th>type_id</th>
      <th>type_name</th>
      <th>possession_team_id</th>
      <th>...</th>
      <th>shot_aerial_won</th>
      <th>foul_committed_penalty</th>
      <th>foul_won_penalty</th>
      <th>50_50_outcome_id</th>
      <th>50_50_outcome_name</th>
      <th>pass_goal_assist</th>
      <th>pass_through_ball</th>
      <th>shot_one_on_one</th>
      <th>foul_committed_offensive</th>
      <th>match_id</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3ac7c9bb-ed06-4a70-9b85-8af22284086b</th>
      <td>1036</td>
      <td>1</td>
      <td>00:26:59.321</td>
      <td>26</td>
      <td>59</td>
      <td>67</td>
      <td>0.680257</td>
      <td>16</td>
      <td>Shot</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>4a74920a-1157-4db1-b657-470acbcead7f</th>
      <td>1298</td>
      <td>1</td>
      <td>00:37:05.297</td>
      <td>37</td>
      <td>5</td>
      <td>84</td>
      <td>0.253100</td>
      <td>16</td>
      <td>Shot</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>292584c4-7c81-4a8a-a512-6944efa9489c</th>
      <td>1318</td>
      <td>1</td>
      <td>00:37:26.659</td>
      <td>37</td>
      <td>26</td>
      <td>84</td>
      <td>1.132100</td>
      <td>16</td>
      <td>Shot</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>953bbd2d-f00a-43ac-a098-816f7644f647</th>
      <td>1394</td>
      <td>1</td>
      <td>00:39:29.724</td>
      <td>39</td>
      <td>29</td>
      <td>90</td>
      <td>0.743200</td>
      <td>16</td>
      <td>Shot</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>fb7ca4cd-2a1b-4231-91d0-e13f3790d34f</th>
      <td>1417</td>
      <td>1</td>
      <td>00:40:30.920</td>
      <td>40</td>
      <td>30</td>
      <td>91</td>
      <td>0.257800</td>
      <td>16</td>
      <td>Shot</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>



## Plot data

As these are football events, we should ideally plot them on a pitch. Borrowing code from [SoccermaticsForPython](https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython/blob/master/FCPython.py), we can first plot the pitch using Matplotlib. The `create_pitch()` function defined in `plot_utils.py` generates the pitch, and it takes pitch length and width as input along with the units of those values. The event data provided by Statsbomb assumes the pitch to be measured in yards.


```python
pitch_length_x: int = 120  # yards
pitch_width_y: int = 80  # yards
```


```python
fig, ax = create_pitch(pitch_length_x, pitch_width_y, "yards", "gray")
```


    
![png](/Users/tejaskale/Code/project-of-the-week-datatalks-club/fastapi-expected-goals/00_loading_investigating_world_cup_data_25_0.png)
    



```python
i: int
shot: dict
for i, shot in shots.iterrows():
    x: int = shot["location"][0]
    y: int = shot["location"][1]

    is_goal: bool = shot["shot_outcome_name"] == "Goal"
    team_name: str = shot["team_name"]

    circle_size: float = np.sqrt(shot["shot_statsbomb_xg"] * 15)

    if team_name == required_home_team:
        shot_circle = plt.Circle((x, pitch_width_y - y), circle_size, color="red")
        if is_goal:
            plt.text((x + 1), (pitch_width_y - y + 1), shot["player_name"])
        else:
            shot_circle.set_alpha(0.2)
    else:
        shot_circle = plt.Circle((pitch_length_x - x, y), circle_size, color="blue")
        if is_goal:
            plt.text((pitch_length_x - x + 1), (y + 1), shot["player_name"])
        else:
            shot_circle.set_alpha(0.2)

    ax.add_patch(shot_circle)

plt.text(5, 75, f"{required_away_team} shots")
plt.text(80, 75, f"{required_home_team} shots")

# fig.set_size_inches(10, 7)
# fig.savefig("results/shots.pdf", dpi=100)
plt.show()
```


    
![png](/Users/tejaskale/Code/project-of-the-week-datatalks-club/fastapi-expected-goals/00_loading_investigating_world_cup_data_26_0.png)
    


Let us now get the data for passes and plot the passes of *Megan Anna Rapinoe* of the USA. When plotting pass maps, it is advisable to plot the passes of one or two players instead of a team as the latter will just lead to a pitch full of arrows from which it will be difficult to derive any meaningful insights.


```python
required_player_name: str = "Megan Anna Rapinoe"
```


```python
passes: pd.DataFrame = events.loc[events["type_name"] == "Pass"].set_index("id")
passes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>period</th>
      <th>timestamp</th>
      <th>minute</th>
      <th>second</th>
      <th>possession</th>
      <th>duration</th>
      <th>type_id</th>
      <th>type_name</th>
      <th>possession_team_id</th>
      <th>...</th>
      <th>shot_aerial_won</th>
      <th>foul_committed_penalty</th>
      <th>foul_won_penalty</th>
      <th>50_50_outcome_id</th>
      <th>50_50_outcome_name</th>
      <th>pass_goal_assist</th>
      <th>pass_through_ball</th>
      <th>shot_one_on_one</th>
      <th>foul_committed_offensive</th>
      <th>match_id</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cda3adf5-db44-4d52-a722-8484b56e9418</th>
      <td>5</td>
      <td>1</td>
      <td>00:00:00.140</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1.213600</td>
      <td>30</td>
      <td>Pass</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>27c0f8c5-e4a3-4461-8b0c-6e90c2fd28da</th>
      <td>8</td>
      <td>1</td>
      <td>00:00:02.589</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2.638200</td>
      <td>30</td>
      <td>Pass</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>b99b019e-ce51-4be7-bd85-0b6d8ffc5b99</th>
      <td>10</td>
      <td>1</td>
      <td>00:00:05.264</td>
      <td>0</td>
      <td>5</td>
      <td>2</td>
      <td>2.215700</td>
      <td>30</td>
      <td>Pass</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>1fa03b06-8ff9-40db-afc5-38117f8e5def</th>
      <td>12</td>
      <td>1</td>
      <td>00:00:07.480</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>1.248558</td>
      <td>30</td>
      <td>Pass</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
    <tr>
      <th>babf6541-d034-40d0-a90f-3472de1404e8</th>
      <td>13</td>
      <td>1</td>
      <td>00:00:13.847</td>
      <td>0</td>
      <td>13</td>
      <td>3</td>
      <td>0.696898</td>
      <td>30</td>
      <td>Pass</td>
      <td>1214</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>69321</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 116 columns</p>
</div>




```python
fig, ax = create_pitch(pitch_length_x, pitch_width_y, "yards", "gray")
```


    
![png](/Users/tejaskale/Code/project-of-the-week-datatalks-club/fastapi-expected-goals/00_loading_investigating_world_cup_data_30_0.png)
    



```python
a_pass: dict  # `pass` is a Python keyword so cannot be used as a variable.
for i, a_pass in passes.iterrows():
    if a_pass["player_name"] != required_player_name:
        continue

    x: int = a_pass["location"][0]
    y: int = a_pass["location"][1]

    pass_circle = plt.Circle((x, pitch_width_y - y), 2, color="blue")
    pass_circle.set_alpha(0.2)

    ax.add_patch(pass_circle)

    dx: int = a_pass["pass_end_location"][0] - x
    dy: int = a_pass["pass_end_location"][1] - y

    pass_arrow = plt.Arrow(x, (pitch_width_y - y), dx, -dy, width=3, color="blue")
    ax.add_patch(pass_arrow)

ax.set_title(f"Passes played by {required_player_name}")
# fig.set_size_inches(10, 7)
# fig.savefig("results/passes.pdf", dpi=100)
plt.show()
```

## References
- [Statsbomb event data](https://github.com/statsbomb/open-data)
- [Wyscout event data](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5)
- [Loading in and investigating World Cup data in Python](https://www.youtube.com/watch?v=GTtuOt03FM0&ab_channel=FriendsofTracking)
- [Making Your Own Shot and Pass Maps](https://www.youtube.com/watch?v=oOAnERLiN5U&ab_channel=FriendsofTracking)


```python

```
