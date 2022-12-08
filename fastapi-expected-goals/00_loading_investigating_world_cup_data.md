# Loading and Investigating World Cup Data

In this notebook, we will understand how to load and inspect event data of Women's World Cup matches. We follow the Prof. David Sumpter's [video](https://www.youtube.com/watch?v=GTtuOt03FM0&ab_channel=FriendsofTracking) for understanding how to download the data and inspect it using Python. During the course of this notebook, we will assume that both Statsbomb and Wyscout data is available in the `data` directory. URLs to download the data are provided in the *References* section.

The event data is provided in JSON files, so we need to import the `json` package to load these files. We will need `matplotlib` to plot the data and `numpy` to transform the data.


```python
import json

import matplotlib.pyplot as plt
import numpy as np
```

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


```python
match: dict
for match in matches:
    home_team_name: str = match["home_team"]["country"]["name"]
    away_team_name: str = match["away_team"]["country"]["name"]
    home_score: int = match["home_score"]
    away_score: int = match["away_score"]
    print(f"The match between {home_team_name} and {away_team_name} finished {home_score}-{away_score}")
```

    The match between Jamaica and Italy finished 0-5
    The match between Jamaica and Australia finished 1-4
    The match between Norway and Australia finished 1-1
    The match between Australia and Italy finished 1-2
    The match between Argentina and Japan finished 0-0
    The match between United States of America and Thailand finished 13-0
    The match between Chile and Sweden finished 0-2
    The match between Nigeria and Korea (South) finished 2-0
    The match between Germany and Spain finished 1-0
    The match between South Africa and China finished 0-1
    The match between Japan and Scotland finished 2-1
    The match between Netherlands and Cameroon finished 3-1
    The match between Canada and New Zealand finished 2-0
    The match between Netherlands and Canada finished 2-1
    The match between Sweden and United States of America finished 0-2
    The match between Sweden and Canada finished 1-0
    The match between Cameroon and New Zealand finished 2-1
    The match between Germany and Nigeria finished 3-0
    The match between England and Sweden finished 1-2
    The match between France and Brazil finished 2-1
    The match between Germany and Sweden finished 1-2
    The match between England and Scotland finished 2-1
    The match between Sweden and Thailand finished 5-1
    The match between Japan and England finished 0-2
    The match between Korea (South) and Norway finished 1-2
    The match between United States of America and Chile finished 3-0
    The match between Norway and Nigeria finished 3-0
    The match between United States of America and Netherlands finished 2-0
    The match between China and Spain finished 0-0
    The match between Italy and Brazil finished 0-1
    The match between France and Korea (South) finished 4-0
    The match between Germany and China finished 1-0
    The match between Spain and South Africa finished 3-1
    The match between Brazil and Jamaica finished 3-0
    The match between Canada and Cameroon finished 1-0
    The match between France and United States of America finished 1-2
    The match between New Zealand and Netherlands finished 0-1
    The match between France and Norway finished 2-1
    The match between Australia and Brazil finished 3-2
    The match between England and Argentina finished 1-0
    The match between Nigeria and France finished 0-1
    The match between South Africa and Germany finished 0-4
    The match between Scotland and Argentina finished 3-3
    The match between Thailand and Chile finished 0-2
    The match between England and Cameroon finished 3-0
    The match between Italy and China finished 2-0
    The match between Netherlands and Japan finished 2-1
    The match between Norway and England finished 0-3
    The match between England and United States of America finished 1-2
    The match between Netherlands and Sweden finished 1-0
    The match between Italy and Netherlands finished 0-2
    The match between Spain and United States of America finished 1-2


Let us consider the final of the World Cup between the USA and Netherlands and find its match ID.


```python
required_home_team: str = "United States of America"
required_away_team: str = "Netherlands"
```


```python
for match in matches:
    home_team_name: str = match["home_team"]["country"]["name"]
    away_team_name: str = match["away_team"]["country"]["name"]
    if (home_team_name == required_home_team) and (away_team_name == required_away_team):
        required_match_id: int = match["match_id"]

print(f"{required_home_team} vs {required_away_team} has ID: {required_match_id}")
```

    United States of America vs Netherlands has ID: 69321


Let us now load the event data for this match based on its ID.


```python
with open(f"./data/statsbomb/data/events/{required_match_id}.json", "r") as f:
    match_events: list = json.load(f)
```

## References
- [Statsbomb event data](https://github.com/statsbomb/open-data)
- [Wyscout event data](https://figshare.com/collections/Soccer_match_event_dataset/4415000/5)
- [Loading in and investigating World Cup data in Python](https://www.youtube.com/watch?v=GTtuOt03FM0&ab_channel=FriendsofTracking)
- [Making Your Own Shot and Pass Maps](https://www.youtube.com/watch?v=oOAnERLiN5U&ab_channel=FriendsofTracking)


```python

```
