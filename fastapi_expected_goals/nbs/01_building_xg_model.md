# Building an Expected Goals Model

In this notebook, we will create a model for computing the probability of a shot being a goal. This probability is referred to as *Expected Goals* (xG) and it is a popular metric in football today to understand how good a team is at creating chances.

Based on the videos by [Prof. David Sumpter](https://uppsala.instructure.com/courses/28112/pages/2-statistical-models-of-actions), we will fit a *Logistic Regression* model to estimate xG. This model will have two input variables - distance of a shot from goal and angle of the shot to the width of the goal. To fit the model, we will use [event data](https://github.com/statsbomb/open-data) from La Liga (Spanish league) matches provided by Statsbomb.

## Imports

To load and inspect this data, we will need the `json` and `pandas` packages. We will need `numpy` for intermediate transformations and `statsmodels` for fitting the Logistic Regression model.


```python
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from mplsoccer import Standardizer
from sklearn.metrics import mean_absolute_error
```

## Data

We have already downloaded the data and placed it in `../data/statsbomb/data` directory. Event data is available per match in a JSON file. So, in order to fetch data for all available La Liga matches, we first need to get the competition ID of La Liga. Let us do so by loading `competitions.json` and inspecting it.


```python
competitions_fn: str = "../data/statsbomb/data/competitions.json"
with open(competitions_fn, "r") as f:
    competitions: pd.DataFrame = pd.read_json(f)
competitions
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
      <th>competition_id</th>
      <th>season_id</th>
      <th>country_name</th>
      <th>competition_name</th>
      <th>competition_gender</th>
      <th>competition_youth</th>
      <th>competition_international</th>
      <th>season_name</th>
      <th>match_updated</th>
      <th>match_updated_360</th>
      <th>match_available_360</th>
      <th>match_available</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>4</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2018/2019</td>
      <td>2022-08-14T16:57:15.866765</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-08-14T16:57:15.866765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>1</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2017/2018</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-01-23T21:55:30.425330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>2</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2016/2017</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>27</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2015/2016</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>16</td>
      <td>26</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2014/2015</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>5</th>
      <td>16</td>
      <td>25</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2013/2014</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>16</td>
      <td>24</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2012/2013</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-07-10T13:41:45.751</td>
    </tr>
    <tr>
      <th>7</th>
      <td>16</td>
      <td>23</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2011/2012</td>
      <td>2021-08-27T11:26:39.802832</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>8</th>
      <td>16</td>
      <td>22</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2010/2011</td>
      <td>2022-01-26T21:07:11.033473</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-01-26T21:07:11.033473</td>
    </tr>
    <tr>
      <th>9</th>
      <td>16</td>
      <td>21</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2009/2010</td>
      <td>2022-02-12T16:13:49.294747</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-02-12T16:13:49.294747</td>
    </tr>
    <tr>
      <th>10</th>
      <td>16</td>
      <td>41</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2008/2009</td>
      <td>2021-11-07T14:20:01.699993</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-11-07T14:20:01.699993</td>
    </tr>
    <tr>
      <th>11</th>
      <td>16</td>
      <td>39</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2006/2007</td>
      <td>2021-03-31T04:18:30.437060</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-03-31T04:18:30.437060</td>
    </tr>
    <tr>
      <th>12</th>
      <td>16</td>
      <td>37</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2004/2005</td>
      <td>2021-04-01T06:18:57.459032</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-04-01T06:18:57.459032</td>
    </tr>
    <tr>
      <th>13</th>
      <td>16</td>
      <td>44</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2003/2004</td>
      <td>2021-04-01T00:34:59.472485</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-04-01T00:34:59.472485</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>76</td>
      <td>Europe</td>
      <td>Champions League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>1999/2000</td>
      <td>2020-07-29T05:00</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>15</th>
      <td>37</td>
      <td>90</td>
      <td>England</td>
      <td>FA Women's Super League</td>
      <td>female</td>
      <td>False</td>
      <td>False</td>
      <td>2020/2021</td>
      <td>2022-08-16T02:10:37.220648</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-08-16T02:10:37.220648</td>
    </tr>
    <tr>
      <th>16</th>
      <td>37</td>
      <td>42</td>
      <td>England</td>
      <td>FA Women's Super League</td>
      <td>female</td>
      <td>False</td>
      <td>False</td>
      <td>2019/2020</td>
      <td>2021-06-01T13:01:18.188</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-06-01T13:01:18.188</td>
    </tr>
    <tr>
      <th>17</th>
      <td>37</td>
      <td>4</td>
      <td>England</td>
      <td>FA Women's Super League</td>
      <td>female</td>
      <td>False</td>
      <td>False</td>
      <td>2018/2019</td>
      <td>2022-09-12T21:06:25.061309</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-09-12T21:06:25.061309</td>
    </tr>
    <tr>
      <th>18</th>
      <td>43</td>
      <td>3</td>
      <td>International</td>
      <td>FIFA World Cup</td>
      <td>male</td>
      <td>False</td>
      <td>True</td>
      <td>2018</td>
      <td>2022-09-05T17:17:56.670896</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-09-05T17:17:56.670896</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1238</td>
      <td>108</td>
      <td>India</td>
      <td>Indian Super league</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2021/2022</td>
      <td>2022-06-08T12:40:59.857067</td>
      <td>None</td>
      <td>None</td>
      <td>2022-06-08T12:40:59.857067</td>
    </tr>
    <tr>
      <th>20</th>
      <td>11</td>
      <td>90</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2020/2021</td>
      <td>2022-02-11T14:56:09.076</td>
      <td>2022-08-16T21:50:36.812060</td>
      <td>2022-08-16T21:50:36.812060</td>
      <td>2022-02-11T14:56:09.076</td>
    </tr>
    <tr>
      <th>21</th>
      <td>11</td>
      <td>42</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2019/2020</td>
      <td>2022-07-15T23:27:24.260122</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-07-15T23:27:24.260122</td>
    </tr>
    <tr>
      <th>22</th>
      <td>11</td>
      <td>4</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2018/2019</td>
      <td>2022-08-30T23:25:57.118855</td>
      <td>2021-07-09T14:53:22.103024</td>
      <td>None</td>
      <td>2022-08-30T23:25:57.118855</td>
    </tr>
    <tr>
      <th>23</th>
      <td>11</td>
      <td>1</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2017/2018</td>
      <td>2022-08-13T23:18:38.928566</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-08-13T23:18:38.928566</td>
    </tr>
    <tr>
      <th>24</th>
      <td>11</td>
      <td>2</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2016/2017</td>
      <td>2022-01-25T22:51:59.353598</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-01-25T22:51:59.353598</td>
    </tr>
    <tr>
      <th>25</th>
      <td>11</td>
      <td>27</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2015/2016</td>
      <td>2022-08-30T13:15:37.855155</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-08-30T13:15:37.855155</td>
    </tr>
    <tr>
      <th>26</th>
      <td>11</td>
      <td>26</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2014/2015</td>
      <td>2022-08-14T18:49:03.341489</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-08-14T18:49:03.341489</td>
    </tr>
    <tr>
      <th>27</th>
      <td>11</td>
      <td>25</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2013/2014</td>
      <td>2022-07-23T12:18:49.547396</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-07-23T12:18:49.547396</td>
    </tr>
    <tr>
      <th>28</th>
      <td>11</td>
      <td>24</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2012/2013</td>
      <td>2022-09-25T20:52:24.444609</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-09-25T20:52:24.444609</td>
    </tr>
    <tr>
      <th>29</th>
      <td>11</td>
      <td>23</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2011/2012</td>
      <td>2020-07-29T05:00</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>30</th>
      <td>11</td>
      <td>22</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2010/2011</td>
      <td>2021-11-11T22:57:42.361902</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-11-11T22:57:42.361902</td>
    </tr>
    <tr>
      <th>31</th>
      <td>11</td>
      <td>21</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2009/2010</td>
      <td>2021-10-26T13:56:40.989214</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-10-26T13:56:40.989214</td>
    </tr>
    <tr>
      <th>32</th>
      <td>11</td>
      <td>41</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2008/2009</td>
      <td>2020-07-29T05:00</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>33</th>
      <td>11</td>
      <td>40</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2007/2008</td>
      <td>2021-10-26T13:13:56.180589</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-10-26T13:13:56.180589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>11</td>
      <td>39</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2006/2007</td>
      <td>2020-07-29T05:00</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>35</th>
      <td>11</td>
      <td>38</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2005/2006</td>
      <td>2022-07-03T12:34:31.749038</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-07-03T12:34:31.749038</td>
    </tr>
    <tr>
      <th>36</th>
      <td>11</td>
      <td>37</td>
      <td>Spain</td>
      <td>La Liga</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2004/2005</td>
      <td>2020-07-29T05:00</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2020-07-29T05:00</td>
    </tr>
    <tr>
      <th>37</th>
      <td>49</td>
      <td>3</td>
      <td>United States of America</td>
      <td>NWSL</td>
      <td>female</td>
      <td>False</td>
      <td>False</td>
      <td>2018</td>
      <td>2021-11-06T05:53:29.435016</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-11-06T05:53:29.435016</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2</td>
      <td>44</td>
      <td>England</td>
      <td>Premier League</td>
      <td>male</td>
      <td>False</td>
      <td>False</td>
      <td>2003/2004</td>
      <td>2021-11-14T22:29:00.646120</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2021-11-14T22:29:00.646120</td>
    </tr>
    <tr>
      <th>39</th>
      <td>55</td>
      <td>43</td>
      <td>Europe</td>
      <td>UEFA Euro</td>
      <td>male</td>
      <td>False</td>
      <td>True</td>
      <td>2020</td>
      <td>2022-02-01T17:20:34.319496</td>
      <td>2022-08-04T12:00</td>
      <td>2022-08-04T12:00</td>
      <td>2022-02-01T17:20:34.319496</td>
    </tr>
    <tr>
      <th>40</th>
      <td>53</td>
      <td>106</td>
      <td>Europe</td>
      <td>UEFA Women's Euro</td>
      <td>female</td>
      <td>False</td>
      <td>True</td>
      <td>2022</td>
      <td>2022-08-01T07:46:11.595364</td>
      <td>2022-09-13T17:04:11.418708</td>
      <td>2022-09-13T17:04:11.418708</td>
      <td>2022-08-01T07:46:11.595364</td>
    </tr>
    <tr>
      <th>41</th>
      <td>72</td>
      <td>30</td>
      <td>International</td>
      <td>Women's World Cup</td>
      <td>female</td>
      <td>False</td>
      <td>True</td>
      <td>2019</td>
      <td>2022-07-13T20:21:27.033445</td>
      <td>2021-06-13T16:17:31.694</td>
      <td>None</td>
      <td>2022-07-13T20:21:27.033445</td>
    </tr>
  </tbody>
</table>
</div>



We see that La Liga has the ID `11`. Let us now load the list of La Liga matches for which event data is available. In `./data/statsbomb/data/matches/11/`, we have one JSON file for every La Liga season. Each of the JSON files provides basic information about all matches in that season. We are interested in getting the ID of each match so that we can load event data for that match from `./data/statsbomb/data/events`. So, let us write function to get IDs of all matches for which we want to get event data.


```python
def get_competition_match_ids(comp_id: int, data_dir: str = "../data/statsbomb/data/") -> list:
    """
    Get IDs of matches from all seasons of a competition e.g. La Liga.
    :param comp_id:  Competition ID from one of those provided in `competitions.json`
    :param data_dir: Path to directory containing `competitions.json` and `matches` directory
    from Statsbomb open data.
    :return: List of match IDs for all seasons.
    """
    comp_dir: str = os.path.join(data_dir, "matches", str(comp_id))
    season_fns: list = os.listdir(comp_dir)
    season_fn: str
    match: dict
    match_ids: list = []
    for season_fn in season_fns:
        with open(os.path.join(comp_dir, season_fn), "r") as jf:
            matches: list = json.load(jf)

        for match in matches:
            match_ids.append(match["match_id"])

    return match_ids
```


```python
required_match_ids: list = get_competition_match_ids(11)
```

Let us write a quick test for our function. Let us pick one match ID at random from the following files:
- `1.json`
- `27.json`
- `42.json`

and verify that our list contains these IDs.


```python
assert 9609 in required_match_ids
assert 266166 in required_match_ids
assert 303532 in required_match_ids
```

Having fetched the match IDs, let us now incrementally load events from them. For our first model, let us only consider shots to goal from open play. To do so, we use our learnings from the [previous exploratory notebook](./00_loading_investigating_world_cup_data.ipynb).


```python
def load_match_events(match_id: int, data_dir: str = "../data/statsbomb/data/events") -> List[dict]:
    """
    Load event data of a match.
    :param match_id: ID of the match which matches the name of the JSON file from which
    to load events
    :param data_dir: Path to the `events` directory of Statsbomb open data.
    :return: A list of dictionaries with each dictionary denoting an event i.e. on-ball action.
    """
    with open(os.path.join(data_dir, f"{match_id}.json"), "r") as jf:
        return json.load(jf)

def frame_events(e: list, match_id: int) -> pd.DataFrame:
    """
    Convert a list of on-ball match events to a Pandas dataframe.
    :param e:        List of dictionaries with each dictionary denoting an event.
    :param match_id: ID of the match whose events are transformed.
    :return: Pandas dataframe of events.
    """
    return (pd.json_normalize(e, sep="_")
            .assign(match_id=match_id))

def filter_events(e: pd.DataFrame, event_type: str, event_type_filter: Optional[dict] = None) -> pd.DataFrame:
    """
    Filter events to include the specified actions. Supported event types are
    one of `["Shot", "Pass"]`. Further filters are supplied as key-value pairs
    with the key of the dictionary interpreted as the column name and the
    dictionary value as the value in the column that will be searched for.
    :param e:                 Dataframe of on-ball match events.
    :param event_type:        Type of events to filter out. Supported values are one
    of `["Shot", "Pass"]`.
    :param event_type_filter: A dictionary specifying filters specific to the event
    specified.
    :return: A Pandas dataframe of events of the specified type.
    """
    if not event_type_filter:
        event_type_filter = {}

    required_event: pd.DataFrame = e.loc[e["type_name"] == event_type].set_index("id")
    col: str
    val: str
    for col, val in event_type_filter.items():
        required_event = required_event.loc[required_event[col] == val]

    return required_event

m_id: int
match_wise_shots: list = []
for m_id in required_match_ids:
    match_events: list = load_match_events(m_id)
    events: pd.DataFrame = frame_events(match_events, m_id)
    match_shots: pd.DataFrame = filter_events(events, "Shot", {"shot_type_name": "Open Play"})
    match_wise_shots.append(match_shots)

la_liga_shots: pd.DataFrame = pd.concat(match_wise_shots)
la_liga_shots
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
      <th>goalkeeper_punched_out</th>
      <th>shot_follows_dribble</th>
      <th>goalkeeper_success_out</th>
      <th>shot_redirect</th>
      <th>half_end_early_video_end</th>
      <th>goalkeeper_lost_in_play</th>
      <th>player_off_permanent</th>
      <th>goalkeeper_saved_to_post</th>
      <th>goalkeeper_success_in_play</th>
      <th>pass_backheel</th>
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
      <th>d0a3a318-9e65-4f79-9a78-1e65d4ea5b82</th>
      <td>104</td>
      <td>1</td>
      <td>00:01:56.139</td>
      <td>1</td>
      <td>56</td>
      <td>7</td>
      <td>1.483300</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>572ebe7b-7efb-4370-98d4-3d178ce0c59d</th>
      <td>247</td>
      <td>1</td>
      <td>00:05:25.322</td>
      <td>5</td>
      <td>25</td>
      <td>15</td>
      <td>0.446500</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>9f17a2a7-976c-4eb7-b4be-7cb6f2b9bdc1</th>
      <td>311</td>
      <td>1</td>
      <td>00:06:29.899</td>
      <td>6</td>
      <td>29</td>
      <td>19</td>
      <td>1.162000</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>bb3d8291-0dba-40a9-9ed0-f395eef891f6</th>
      <td>662</td>
      <td>1</td>
      <td>00:15:56.988</td>
      <td>15</td>
      <td>56</td>
      <td>34</td>
      <td>0.616861</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>71faa4ab-6734-4fa1-b309-29df1020285c</th>
      <td>684</td>
      <td>1</td>
      <td>00:16:30.980</td>
      <td>16</td>
      <td>30</td>
      <td>37</td>
      <td>0.174000</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6653337b-5599-453f-872d-379cfb5823eb</th>
      <td>3664</td>
      <td>2</td>
      <td>00:34:48.656</td>
      <td>79</td>
      <td>48</td>
      <td>200</td>
      <td>1.087292</td>
      <td>16</td>
      <td>Shot</td>
      <td>215</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>7afa362d-de22-4a37-ac93-7d2f7579c593</th>
      <td>3730</td>
      <td>2</td>
      <td>00:36:38.351</td>
      <td>81</td>
      <td>38</td>
      <td>204</td>
      <td>0.293700</td>
      <td>16</td>
      <td>Shot</td>
      <td>215</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>36ec2b3a-e40c-40bb-8c60-fc92bebf3a93</th>
      <td>3885</td>
      <td>2</td>
      <td>00:40:02.956</td>
      <td>85</td>
      <td>2</td>
      <td>210</td>
      <td>0.549300</td>
      <td>16</td>
      <td>Shot</td>
      <td>215</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>8645c39f-b071-4b79-ba36-ab433ef4d79b</th>
      <td>4036</td>
      <td>2</td>
      <td>00:43:42.852</td>
      <td>88</td>
      <td>42</td>
      <td>218</td>
      <td>0.845001</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
    <tr>
      <th>6ea4aa64-c880-40fb-94d6-970fd1925949</th>
      <td>4088</td>
      <td>2</td>
      <td>00:45:21.080</td>
      <td>90</td>
      <td>21</td>
      <td>223</td>
      <td>0.840600</td>
      <td>16</td>
      <td>Shot</td>
      <td>217</td>
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
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>11700 rows × 147 columns</p>
</div>



## Baseline model

Let us now construct a baseline Logistic Regression model. To do so, we require the following information:
- X and Y coordinates from where the shot was taken
- Whether the shot resulted in a goal

From the X and Y coordinates, we will next create two more columns - one to compute the distance of the shot from goal and the other to compute the angle of the shot to the goal.

Looking at the data above, we see a column named `shot_statsbomb_xg`. We can use this column as a reference for our model results, but we won't consider those values to be the ground truth.

The X and Y coordinates are available as a list in the `location` column. We will need to extract them and put them in separate columns. The column `shot_outcome_id` tells us the outcome of the shot. Based on page 20 of the document `./data/statsbomb/doc/Open Data Events v4.0.0.pdf`, we can see that shots with `shot_outcome_id = 97` are goals while the others are not. So, we need to construct a boolean column from it accordingly. Let us define a function that will perform these steps.

In addition, the [tutorial](https://www.youtube.com/watch?v=wHOgINJ5g54) we follow for building this model uses Wyscout data and the two data providers measure the pitch and thus record shot coordinates in different units. As we will use the logic used for Wyscout data to compute shot distance and angle, we make use of the `Standardizer` class of `mplsoccer` package to transform our coordinates from Statsbomb to Wyscout units.


```python
def create_shot_modelling_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform event data of shots for modelling. This includes:
    - Splitting the `location` column into two - one for representing X coordinate and the other Y.
    - Converting the coordinate values to Wyscout units as further computations assume
    the coordinates to be in Wyscout units.
    - Creating a boolean column representing if the shot resulted in a goal (1) or not (0).
    - Dropping all column except the coordinates, boolean indicator of goal, and Statsbomb's xG value.
    - Renaming Statsbomb's xG value column.
    :param df: Pandas dataframe of event data for shots.
    :return: A Pandas dataframe with columns - X and Y coordinate of the shot, boolean indicator of goal,
    and Statsbomb's xG.
    """
    statsbomb_to_wyscout = Standardizer(pitch_from="statsbomb", pitch_to="wyscout")
    return (df.assign(X=lambda x: [l[0] for l in x["location"]])
            .assign(Y=lambda x: [l[1] for l in x["location"]])
            .assign(X=lambda x: [round(statsbomb_to_wyscout.transform([xi], [yi])[0][0], 2)
                                 for xi, yi in zip(x["X"], x["Y"])])
            .assign(Y=lambda x: [round(statsbomb_to_wyscout.transform([xi], [yi])[1][0], 2)
                                 for xi, yi in zip(x["X"], x["Y"])])
            .assign(is_goal=lambda x: (x["shot_outcome_id"] == 97).astype(int))
            .filter(["X", "Y", "is_goal", "shot_statsbomb_xg"], axis=1)
            .rename(columns={"shot_statsbomb_xg": "statsbomb_xg"}))

la_liga_shots_model_data: pd.DataFrame = create_shot_modelling_data(la_liga_shots)
la_liga_shots_model_data
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
      <th>X</th>
      <th>Y</th>
      <th>is_goal</th>
      <th>statsbomb_xg</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d0a3a318-9e65-4f79-9a78-1e65d4ea5b82</th>
      <td>69.02</td>
      <td>35.80</td>
      <td>0</td>
      <td>0.010060</td>
    </tr>
    <tr>
      <th>572ebe7b-7efb-4370-98d4-3d178ce0c59d</th>
      <td>91.60</td>
      <td>69.30</td>
      <td>0</td>
      <td>0.062470</td>
    </tr>
    <tr>
      <th>9f17a2a7-976c-4eb7-b4be-7cb6f2b9bdc1</th>
      <td>76.88</td>
      <td>28.15</td>
      <td>0</td>
      <td>0.012832</td>
    </tr>
    <tr>
      <th>bb3d8291-0dba-40a9-9ed0-f395eef891f6</th>
      <td>90.07</td>
      <td>67.65</td>
      <td>0</td>
      <td>0.043368</td>
    </tr>
    <tr>
      <th>71faa4ab-6734-4fa1-b309-29df1020285c</th>
      <td>82.95</td>
      <td>44.90</td>
      <td>0</td>
      <td>0.144784</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6653337b-5599-453f-872d-379cfb5823eb</th>
      <td>79.06</td>
      <td>57.40</td>
      <td>0</td>
      <td>0.023162</td>
    </tr>
    <tr>
      <th>7afa362d-de22-4a37-ac93-7d2f7579c593</th>
      <td>90.93</td>
      <td>33.70</td>
      <td>0</td>
      <td>0.110399</td>
    </tr>
    <tr>
      <th>36ec2b3a-e40c-40bb-8c60-fc92bebf3a93</th>
      <td>83.11</td>
      <td>23.05</td>
      <td>0</td>
      <td>0.012569</td>
    </tr>
    <tr>
      <th>8645c39f-b071-4b79-ba36-ab433ef4d79b</th>
      <td>92.33</td>
      <td>32.20</td>
      <td>0</td>
      <td>0.068940</td>
    </tr>
    <tr>
      <th>6ea4aa64-c880-40fb-94d6-970fd1925949</th>
      <td>81.73</td>
      <td>60.55</td>
      <td>0</td>
      <td>0.229089</td>
    </tr>
  </tbody>
</table>
<p>11700 rows × 4 columns</p>
</div>



Let us now compute shot distance and angle.


```python
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

la_liga_xg_model_data: pd.DataFrame = (la_liga_shots_model_data
                                       .assign(dist=lambda x: [compute_shot_distance(xc, yc)
                                                               for xc, yc in zip(x["X"], x["Y"])])
                                       .assign(angle=lambda x: [compute_shot_angle(xc, yc)
                                                                for xc, yc in zip(x["X"], x["Y"])]))
la_liga_xg_model_data
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
      <th>X</th>
      <th>Y</th>
      <th>is_goal</th>
      <th>statsbomb_xg</th>
      <th>dist</th>
      <th>angle</th>
    </tr>
    <tr>
      <th>id</th>
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
      <th>d0a3a318-9e65-4f79-9a78-1e65d4ea5b82</th>
      <td>69.02</td>
      <td>35.80</td>
      <td>0</td>
      <td>0.010060</td>
      <td>33.813145</td>
      <td>0.207693</td>
    </tr>
    <tr>
      <th>572ebe7b-7efb-4370-98d4-3d178ce0c59d</th>
      <td>91.60</td>
      <td>69.30</td>
      <td>0</td>
      <td>0.062470</td>
      <td>15.335235</td>
      <td>0.283289</td>
    </tr>
    <tr>
      <th>9f17a2a7-976c-4eb7-b4be-7cb6f2b9bdc1</th>
      <td>76.88</td>
      <td>28.15</td>
      <td>0</td>
      <td>0.012832</td>
      <td>28.125348</td>
      <td>0.224655</td>
    </tr>
    <tr>
      <th>bb3d8291-0dba-40a9-9ed0-f395eef891f6</th>
      <td>90.07</td>
      <td>67.65</td>
      <td>0</td>
      <td>0.043368</td>
      <td>15.502586</td>
      <td>0.324434</td>
    </tr>
    <tr>
      <th>71faa4ab-6734-4fa1-b309-29df1020285c</th>
      <td>82.95</td>
      <td>44.90</td>
      <td>0</td>
      <td>0.144784</td>
      <td>18.206832</td>
      <td>0.390787</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6653337b-5599-453f-872d-379cfb5823eb</th>
      <td>79.06</td>
      <td>57.40</td>
      <td>0</td>
      <td>0.023162</td>
      <td>22.506983</td>
      <td>0.315451</td>
    </tr>
    <tr>
      <th>7afa362d-de22-4a37-ac93-7d2f7579c593</th>
      <td>90.93</td>
      <td>33.70</td>
      <td>0</td>
      <td>0.110399</td>
      <td>14.246090</td>
      <td>0.352413</td>
    </tr>
    <tr>
      <th>36ec2b3a-e40c-40bb-8c60-fc92bebf3a93</th>
      <td>83.11</td>
      <td>23.05</td>
      <td>0</td>
      <td>0.012569</td>
      <td>24.927401</td>
      <td>0.210362</td>
    </tr>
    <tr>
      <th>8645c39f-b071-4b79-ba36-ab433ef4d79b</th>
      <td>92.33</td>
      <td>32.20</td>
      <td>0</td>
      <td>0.068940</td>
      <td>14.096942</td>
      <td>0.307972</td>
    </tr>
    <tr>
      <th>6ea4aa64-c880-40fb-94d6-970fd1925949</th>
      <td>81.73</td>
      <td>60.55</td>
      <td>0</td>
      <td>0.229089</td>
      <td>20.372334</td>
      <td>0.336343</td>
    </tr>
  </tbody>
</table>
<p>11700 rows × 6 columns</p>
</div>



Let us now fit a Logistic Regression model using statsmodel's `glm()` method. As our output is binary, we specify the `family` argument to be `sm.families.Binomial()`. After fitting the model, let us print a summary of the model.


```python
baseline_model = smf.glm(formula="is_goal ~ dist + angle", data=la_liga_xg_model_data,
                         family=sm.families.Binomial()).fit()
baseline_model.summary()
```




<table class="simpletable">
<caption>Generalized Linear Model Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>is_goal</td>     <th>  No. Observations:  </th>  <td> 11700</td> 
</tr>
<tr>
  <th>Model:</th>                  <td>GLM</td>       <th>  Df Residuals:      </th>  <td> 11697</td> 
</tr>
<tr>
  <th>Model Family:</th>        <td>Binomial</td>     <th>  Df Model:          </th>  <td>     2</td> 
</tr>
<tr>
  <th>Link Function:</th>         <td>Logit</td>      <th>  Scale:             </th> <td>  1.0000</td>
</tr>
<tr>
  <th>Method:</th>                <td>IRLS</td>       <th>  Log-Likelihood:    </th> <td> -4024.5</td>
</tr>
<tr>
  <th>Date:</th>            <td>Sat, 10 Dec 2022</td> <th>  Deviance:          </th> <td>  8049.0</td>
</tr>
<tr>
  <th>Time:</th>                <td>22:33:31</td>     <th>  Pearson chi2:      </th> <td>1.15e+04</td>
</tr>
<tr>
  <th>No. Iterations:</th>          <td>6</td>        <th>  Pseudo R-squ. (CS):</th>  <td>0.08902</td>
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   -1.2611</td> <td>    0.187</td> <td>   -6.739</td> <td> 0.000</td> <td>   -1.628</td> <td>   -0.894</td>
</tr>
<tr>
  <th>dist</th>      <td>   -0.0936</td> <td>    0.008</td> <td>  -11.455</td> <td> 0.000</td> <td>   -0.110</td> <td>   -0.078</td>
</tr>
<tr>
  <th>angle</th>     <td>    1.4011</td> <td>    0.162</td> <td>    8.636</td> <td> 0.000</td> <td>    1.083</td> <td>    1.719</td>
</tr>
</table>



Looking at the model summary above, in particular the last table, we see that the probability of a shot becoming a goal decreases with increasing distance. We say this based on the negative value of the coefficient. Similarly, as the shot angle increases, i.e. as the shot is taken from between the goal posts, the probability of hitting the back of the net also increases. The near-zero P-values of both the coefficients suggest that we have sufficient evidence to reject the null hypothesis that the true coefficient value is zero.

Let us now save the model parameters in a `.pkl` file.


```python
def save_model(params: pd.Series, model_fn: str, model_dir: str = "../models"):
    """
    Save parameters of Logistic Regression to a pickle file.
    :param params:    A Pandas Series of Logistic Regression parameters.
    :param model_fn:  Name of pickle file (with extension) to save the parameters to.
    :param model_dir: Directory to save the model to.
    """
    with open(os.path.join(model_dir, model_fn), "wb") as pf:
        pickle.dump(params, pf)

baseline_model_params: pd.Series = baseline_model.params
baseline_model_fn: str = "baseline_logistic_model.pkl"
save_model(baseline_model_params, baseline_model_fn)
```

## Serving the model

Let us now define a function that loads the parameters from a pickle file and computes the xG value.


```python
def compute_xg(xc: float, yc: float, model_fn: str, model_dir: str = "../models") -> float:
    """
    Compute the xG of a shot given the (X, Y) coordinates of where it was taken from.
    :param xc:        X-coordinate of where the shot was taken from.
    :param yc:        Y-coordinate of where the shot was taken from.
    :param model_fn:  Name of the file containing Logistic Regression parameters.
    :param model_dir: Directory containing the model file.
    :return: A float representing xG value.
    """
    shot_dist: float = compute_shot_distance(xc, yc)
    shot_ang: float = compute_shot_angle(xc, yc)

    with open(os.path.join(model_dir, model_fn), "rb") as pf:
        model_params: pd.Series = pickle.load(pf)

    linear_sum: float = (model_params["Intercept"]
                         + (shot_dist * model_params["dist"])
                         + (shot_ang * model_params["angle"]))

    return 1 / (1 + np.exp(-1 * linear_sum))

la_liga_xg_model_data = (la_liga_xg_model_data
                         .assign(xg=lambda x: [compute_xg(xi, yi, baseline_model_fn) for xi, yi in zip(x["X"], x["Y"])]))
la_liga_xg_model_data
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
      <th>X</th>
      <th>Y</th>
      <th>is_goal</th>
      <th>statsbomb_xg</th>
      <th>dist</th>
      <th>angle</th>
      <th>xg</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>d0a3a318-9e65-4f79-9a78-1e65d4ea5b82</th>
      <td>69.02</td>
      <td>35.80</td>
      <td>0</td>
      <td>0.010060</td>
      <td>33.813145</td>
      <td>0.207693</td>
      <td>0.015755</td>
    </tr>
    <tr>
      <th>572ebe7b-7efb-4370-98d4-3d178ce0c59d</th>
      <td>91.60</td>
      <td>69.30</td>
      <td>0</td>
      <td>0.062470</td>
      <td>15.335235</td>
      <td>0.283289</td>
      <td>0.091168</td>
    </tr>
    <tr>
      <th>9f17a2a7-976c-4eb7-b4be-7cb6f2b9bdc1</th>
      <td>76.88</td>
      <td>28.15</td>
      <td>0</td>
      <td>0.012832</td>
      <td>28.125348</td>
      <td>0.224655</td>
      <td>0.027156</td>
    </tr>
    <tr>
      <th>bb3d8291-0dba-40a9-9ed0-f395eef891f6</th>
      <td>90.07</td>
      <td>67.65</td>
      <td>0</td>
      <td>0.043368</td>
      <td>15.502586</td>
      <td>0.324434</td>
      <td>0.094707</td>
    </tr>
    <tr>
      <th>71faa4ab-6734-4fa1-b309-29df1020285c</th>
      <td>82.95</td>
      <td>44.90</td>
      <td>0</td>
      <td>0.144784</td>
      <td>18.206832</td>
      <td>0.390787</td>
      <td>0.081841</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6653337b-5599-453f-872d-379cfb5823eb</th>
      <td>79.06</td>
      <td>57.40</td>
      <td>0</td>
      <td>0.023162</td>
      <td>22.506983</td>
      <td>0.315451</td>
      <td>0.050903</td>
    </tr>
    <tr>
      <th>7afa362d-de22-4a37-ac93-7d2f7579c593</th>
      <td>90.93</td>
      <td>33.70</td>
      <td>0</td>
      <td>0.110399</td>
      <td>14.246090</td>
      <td>0.352413</td>
      <td>0.109032</td>
    </tr>
    <tr>
      <th>36ec2b3a-e40c-40bb-8c60-fc92bebf3a93</th>
      <td>83.11</td>
      <td>23.05</td>
      <td>0</td>
      <td>0.012569</td>
      <td>24.927401</td>
      <td>0.210362</td>
      <td>0.035593</td>
    </tr>
    <tr>
      <th>8645c39f-b071-4b79-ba36-ab433ef4d79b</th>
      <td>92.33</td>
      <td>32.20</td>
      <td>0</td>
      <td>0.068940</td>
      <td>14.096942</td>
      <td>0.307972</td>
      <td>0.104427</td>
    </tr>
    <tr>
      <th>6ea4aa64-c880-40fb-94d6-970fd1925949</th>
      <td>81.73</td>
      <td>60.55</td>
      <td>0</td>
      <td>0.229089</td>
      <td>20.372334</td>
      <td>0.336343</td>
      <td>0.063178</td>
    </tr>
  </tbody>
</table>
<p>11700 rows × 7 columns</p>
</div>



From the dataframe above, we see that our computed xG differs from the Statsbomb one. If we were to measure the difference in absolute terms, we see that the two xG values differ by about 7%.


```python
mean_absolute_error(la_liga_xg_model_data["statsbomb_xg"], la_liga_xg_model_data["xg"])
```




    0.071261113252593



## Improvements

If we look at the documentation of event information in `./data/statsbomb/doc/Open Data Events v4.0.0.pdf`, we can spot additional parameters that might improve the model. These include:
- Freeze-frame which tells us about the opposition players in the vicinity when a shot was taken. This can be an important variable as more players around means more pressure on the player which in turn can lead to a false shot.
- Open goal which tells us if the shot was taken in front of an open goal.
- Deflected which tells us if the shot was deflected.
- Technique whose values can be one of Backheel, Diving header, Half volley, Lob, Normal, Overhead kick, or volley.
- Body part which indicates if the shot was taken with the head, left foot, right foot, or other body part. This variable combined with information about which foot a player prefers can be useful in predicting if a shot will turn into a goal.

In his video [The Ultimate Guide to Expected Goals](https://www.youtube.com/watch?v=310_eW0hUqQ), Prof. Sumpter asserts that given the right variables, models more complex than Logistic Regression might not provide much better performance. Thus, improvements to the model can focus on adding more variables first and evaluating if they contribute to the model fit before experimenting with other models.


```python

```
