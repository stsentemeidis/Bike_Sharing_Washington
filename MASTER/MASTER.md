
# Bike Sharing in Washington D.C.

Statistical Programming - Python | MBD OCT 2018 | O17 (Group G)  
*IE School of Human Sciences and Technology*  

***

## Introduction

### Objectives

This case study of the Washington D.C Bike Sharing System aims to predict the total number of users on an hourly basis. The dataset is [available on Kaggle](https://www.kaggle.com/marklvl/bike-sharing-dataset/home). It contains usage information of years 2011 and 2012.

All the files of this project are saved in a [GitHub repository](https://github.com/ashomah/Bike-Sharing-in-Washington).

### Libraries

This project uses a set of libraries for data manipulation, ploting and modelling.


```python
# Loading Libraries
import pandas as pd #Data Manipulation - version 0.23.4
pd.set_option('display.max_columns', 500)
import numpy as np #Data Manipulation - version 1.15.4
import datetime

import matplotlib.pyplot as plt #Plotting - version 3.0.2
import matplotlib.ticker as ticker #Plotting - version 3.0.2
import seaborn as sns #Plotting - version 0.9.0
sns.set(style='white')

from sklearn import preprocessing #Preprocessing - version 0.20.1
from sklearn.preprocessing import MinMaxScaler #Preprocessing - version 0.20.1
from sklearn.preprocessing import PolynomialFeatures #Preprocessing - version 0.20.1

from scipy.stats import skew, boxcox_normmax #Preprocessing - version 1.1.0
from scipy.special import boxcox1p #Preprocessing - version 1.1.0
import statsmodels.api as sm #Outliers detection - version 0.9.0

from sklearn.model_selection import train_test_split #Train/Test Split - version 0.20.1
from sklearn.model_selection import TimeSeriesSplit,cross_validate #Timeseries CV - version 0.20.1
from sklearn import datasets, linear_model #Model - version 0.20.1
from sklearn.linear_model import LinearRegression #Model - version 0.20.1

from sklearn.metrics import mean_squared_error, r2_score #Metrics - version 0.20.1
from sklearn.metrics import accuracy_score #Metrics - version 0.20.1
from sklearn.model_selection import cross_val_score, cross_val_predict # CV - version 0.20.1
from sklearn.feature_selection import RFE #Feature Selection - version 0.20.1
```

### Data Loading

The dataset is stored in the [GitHub repository](https://github.com/ashomah/Bike-Sharing-in-Washington) consisting in two CSV file: `day.csv` and `hour.csv`. The files are loaded directly from the repository.


```python
hours_df = pd.read_csv("https://raw.githubusercontent.com/ashomah/Bike-Sharing-in-Washington/master/Bike-Sharing-Dataset/hour.csv")
hours_df.head()
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
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
days_df = pd.read_csv("https://raw.githubusercontent.com/ashomah/Bike-Sharing-in-Washington/master/Bike-Sharing-Dataset/day.csv")
days_df.head()
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
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.344167</td>
      <td>0.363625</td>
      <td>0.805833</td>
      <td>0.160446</td>
      <td>331</td>
      <td>654</td>
      <td>985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.363478</td>
      <td>0.353739</td>
      <td>0.696087</td>
      <td>0.248539</td>
      <td>131</td>
      <td>670</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.196364</td>
      <td>0.189405</td>
      <td>0.437273</td>
      <td>0.248309</td>
      <td>120</td>
      <td>1229</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.200000</td>
      <td>0.212122</td>
      <td>0.590435</td>
      <td>0.160296</td>
      <td>108</td>
      <td>1454</td>
      <td>1562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-05</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.226957</td>
      <td>0.229270</td>
      <td>0.436957</td>
      <td>0.186900</td>
      <td>82</td>
      <td>1518</td>
      <td>1600</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preparation

### Variables Types and Definitions

The first stage of this analysis is to describe the dataset, understand the meaning of variable and perform the necessary adjustments to ensure that the data will be proceeded correctly during the Machine Learning process.


```python
# Shape of the data frame
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('hour.csv:', hours_df.shape[0],'rows |', hours_df.shape[1], 'columns'))
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('day.csv:', days_df.shape[0],'rows |', days_df.shape[1], 'columns'))
```

    hour.csv:  17379 rows |  17 columns
    day.csv:     731 rows |  16 columns



```python
# Describe each variable
def df_desc(df):
    import pandas as pd
    desc = pd.DataFrame({'dtype': df.dtypes,
                         'NAs': df.isna().sum(),
                         'Numerical': (df.dtypes != 'object') & (df.dtypes != 'datetime64[ns]') & (df.apply(lambda column: column == 0).sum() + df.apply(lambda column: column == 1).sum() != len(df)),
                         'Boolean': df.apply(lambda column: column == 0).sum() + df.apply(lambda column: column == 1).sum() == len(df),
                         'Categorical': df.dtypes == 'object',
                         'Date': df.dtypes == 'datetime64[ns]',
                        })
    return desc
```


```python
df_desc(days_df)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>instant</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>dteday</th>
      <td>object</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>yr</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mnth</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weathersit</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hum</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>cnt</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_desc(hours_df)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>instant</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>dteday</th>
      <td>object</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>yr</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>mnth</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hr</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weathersit</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hum</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>cnt</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



The dataset `day.csv` consists in 731 rows and 16 columns. The dataset `hour.csv` consists in 17,379 rows and 17 columns. Both datasets have the same columns, with an additional column for hours in `hour.csv`.

Each row provides information for each day or each hour. None of the attributes contains any NA. Four (4) of these attributes contain decimal numbers, nine (9) contain integers, three (3) contain booleans, and one (1) contains date values stored as string.

For better readability, the columns of both data frames are renamed and data types are adjusted.


```python
# HOURS DATASET
# Renaming columns names to more readable names
hours_df.rename(columns={'instant':'id',
                        'dteday':'date',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_bikes',
                        'hr':'hour',
                        'yr':'year',
                        'temp':'actual_temp',
                        'atemp':'feeling_temp'},
                inplace=True)

# Date time conversion
hours_df.date = pd.to_datetime(hours_df.date, format='%Y-%m-%d')

# Categorical variables
for column in ['season', 'holiday', 'weekday', 'workingday', 'weather_condition','month', 'year','hour']:
    hours_df[column] = hours_df[column].astype('category')
    
# DAYS DATASET
# Renaming columns names to more readable names
days_df.rename(columns={'instant':'id',
                        'dteday':'date',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_bikes',
                        'yr':'year',
                        'temp':'actual_temp',
                        'atemp':'feeling_temp'},
               inplace=True)

# Date time conversion
days_df.date = pd.to_datetime(days_df.date, format='%Y-%m-%d')

# Categorical variables
for column in ['season', 'holiday', 'weekday', 'workingday', 'weather_condition','month', 'year']:
    days_df[column] = days_df[column].astype('category')
```


```python
hours_df.head()
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
      <th>date</th>
      <th>season</th>
      <th>year</th>
      <th>month</th>
      <th>hour</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weather_condition</th>
      <th>actual_temp</th>
      <th>feeling_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
hours_df.describe()
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
      <th>actual_temp</th>
      <th>feeling_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17379.0000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8690.0000</td>
      <td>0.496987</td>
      <td>0.475775</td>
      <td>0.627229</td>
      <td>0.190098</td>
      <td>35.676218</td>
      <td>153.786869</td>
      <td>189.463088</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5017.0295</td>
      <td>0.192556</td>
      <td>0.171850</td>
      <td>0.192930</td>
      <td>0.122340</td>
      <td>49.305030</td>
      <td>151.357286</td>
      <td>181.387599</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.0000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4345.5000</td>
      <td>0.340000</td>
      <td>0.333300</td>
      <td>0.480000</td>
      <td>0.104500</td>
      <td>4.000000</td>
      <td>34.000000</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8690.0000</td>
      <td>0.500000</td>
      <td>0.484800</td>
      <td>0.630000</td>
      <td>0.194000</td>
      <td>17.000000</td>
      <td>115.000000</td>
      <td>142.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>13034.5000</td>
      <td>0.660000</td>
      <td>0.621200</td>
      <td>0.780000</td>
      <td>0.253700</td>
      <td>48.000000</td>
      <td>220.000000</td>
      <td>281.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>17379.0000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.850700</td>
      <td>367.000000</td>
      <td>886.000000</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lists values of categorical variables
categories = {'season': hours_df['season'].unique().tolist(),
              'year':hours_df['year'].unique().tolist(),
              'month':hours_df['month'].unique().tolist(),
              'hour':hours_df['hour'].unique().tolist(),
              'holiday':hours_df['holiday'].unique().tolist(),
              'weekday':hours_df['weekday'].unique().tolist(),
              'workingday':hours_df['workingday'].unique().tolist(),
              'weather_condition':hours_df['weather_condition'].unique().tolist(),
             }
for i in sorted(categories.keys()):
    print(i+":")
    print(categories[i])
    if i != sorted(categories.keys())[-1] :print()
```

    holiday:
    [0, 1]
    
    hour:
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    month:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    season:
    [1, 2, 3, 4]
    
    weather_condition:
    [1, 2, 3, 4]
    
    weekday:
    [6, 0, 1, 2, 3, 4, 5]
    
    workingday:
    [0, 1]
    
    year:
    [0, 1]



```python
df_desc(hours_df)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>date</th>
      <td>datetime64[ns]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>season</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>feeling_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
days_df.head()
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
      <th>date</th>
      <th>season</th>
      <th>year</th>
      <th>month</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weather_condition</th>
      <th>actual_temp</th>
      <th>feeling_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0.344167</td>
      <td>0.363625</td>
      <td>0.805833</td>
      <td>0.160446</td>
      <td>331</td>
      <td>654</td>
      <td>985</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-02</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0.363478</td>
      <td>0.353739</td>
      <td>0.696087</td>
      <td>0.248539</td>
      <td>131</td>
      <td>670</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-03</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0.196364</td>
      <td>0.189405</td>
      <td>0.437273</td>
      <td>0.248309</td>
      <td>120</td>
      <td>1229</td>
      <td>1349</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-04</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0.200000</td>
      <td>0.212122</td>
      <td>0.590435</td>
      <td>0.160296</td>
      <td>108</td>
      <td>1454</td>
      <td>1562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-05</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.226957</td>
      <td>0.229270</td>
      <td>0.436957</td>
      <td>0.186900</td>
      <td>82</td>
      <td>1518</td>
      <td>1600</td>
    </tr>
  </tbody>
</table>
</div>




```python
days_df.describe()
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
      <th>actual_temp</th>
      <th>feeling_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
      <td>731.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>366.000000</td>
      <td>0.495385</td>
      <td>0.474354</td>
      <td>0.627894</td>
      <td>0.190486</td>
      <td>848.176471</td>
      <td>3656.172367</td>
      <td>4504.348837</td>
    </tr>
    <tr>
      <th>std</th>
      <td>211.165812</td>
      <td>0.183051</td>
      <td>0.162961</td>
      <td>0.142429</td>
      <td>0.077498</td>
      <td>686.622488</td>
      <td>1560.256377</td>
      <td>1937.211452</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.059130</td>
      <td>0.079070</td>
      <td>0.000000</td>
      <td>0.022392</td>
      <td>2.000000</td>
      <td>20.000000</td>
      <td>22.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>183.500000</td>
      <td>0.337083</td>
      <td>0.337842</td>
      <td>0.520000</td>
      <td>0.134950</td>
      <td>315.500000</td>
      <td>2497.000000</td>
      <td>3152.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>366.000000</td>
      <td>0.498333</td>
      <td>0.486733</td>
      <td>0.626667</td>
      <td>0.180975</td>
      <td>713.000000</td>
      <td>3662.000000</td>
      <td>4548.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>548.500000</td>
      <td>0.655417</td>
      <td>0.608602</td>
      <td>0.730209</td>
      <td>0.233214</td>
      <td>1096.000000</td>
      <td>4776.500000</td>
      <td>5956.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>731.000000</td>
      <td>0.861667</td>
      <td>0.840896</td>
      <td>0.972500</td>
      <td>0.507463</td>
      <td>3410.000000</td>
      <td>6946.000000</td>
      <td>8714.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lists values of categorical variables
categories = {'season': days_df['season'].unique().tolist(),
              'year':days_df['year'].unique().tolist(),
              'month':days_df['month'].unique().tolist(),
              'holiday':days_df['holiday'].unique().tolist(),
              'weekday':days_df['weekday'].unique().tolist(),
              'workingday':days_df['workingday'].unique().tolist(),
              'weather_condition':days_df['weather_condition'].unique().tolist(),
             }
for i in sorted(categories.keys()):
    print(i+":")
    print(categories[i])
    if i != sorted(categories.keys())[-1] :print()
```

    holiday:
    [0, 1]
    
    month:
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    season:
    [1, 2, 3, 4]
    
    weather_condition:
    [2, 1, 3]
    
    weekday:
    [6, 0, 1, 2, 3, 4, 5]
    
    workingday:
    [0, 1]
    
    year:
    [0, 1]



```python
df_desc(days_df)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>date</th>
      <td>datetime64[ns]</td>
      <td>0</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>season</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>category</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition</th>
      <td>category</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>feeling_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>int64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



For this study, we will only work with the dataset `hours`. The datasets contain 17 variables with no NAs:

- `id`: numerical, integer values.  
  *Record index. __This variable won't be considered in the study.__*
  
  
- `date`: numerical, date values.  
  *Date.*


- `season`: encoded categorical, integer between 1 and 4.  
  *Season: 2=Spring, 3=Summer, 4=Fall, 1=Winter.*  
  *__Note: the seasons mentioned on the Kaggle page didn't correspond to the real seasons. We readjusted the parameters accordingly.__*


- `year`: encoded categorical, integer between 0 and 1.  
  *Year: 0=2011, 1=2012.*
  
  
- `month`: encoded categorical, integer between 1 and 12.  
  *Month.*
  
  
- `hour`: encoded categorical, integer between 1 and 23.  
  *Hour.*
  
  
- `holiday`: encoded categorical, boolean.  
  *Flag indicating if the day is a holiday.*


- `weekday`: encoded categorical, integer between 0 and 6.  
  *Day of the week (0=Sunday, ... 6=Saturday).*


- `workingday`: encoded categorical, boolean.  
  *Flag indicating if the day is a working day.*
  
  
- `weather_condition`: encoded categorical, integer between 1 and 4.  
  *Weather condition (1=Clear, 2=Mist, 3=Light Rain, 4=Heavy Rain).*


- `actual_temp`: numerical, decimal values between 0 and 1.  
  *Normalized temperature in Celsius (min = -16, max = +50).*


- `feeling_temp`: numerical, decimal values between 0 and 1.  
  *Normalized feeling temperature in Celsius (min = -8, max = +39).*


- `humidity`: numerical, decimal values between 0 and 1.  
  *Normalized humidity.*


- `windspeed`: numerical, decimal values between 0 and 1.  
  *Normalized wind speed.*


- `casual`: numerical, integer.  
  *Count of casual users. This variable won't be considered in the study.*


- `registered`: numerical, integer.  
  *Count of registered users. This variable won't be considered in the study.*


- `total_bikes`: numerical, integer.  
  *Count of total rental bikes (casual+registered). This is the __target variable__ of the study, the one to be modelled.*


```python
# Remove variable id
hours_df= hours_df.drop(['id'], axis=1)
```

### Exploratory Data Analysis

#### Bike sharing utilization over the two years

The objective of this study is to build a model to predict the value of the variable `total_bikes`, based on the other variables available.


```python
# Total_bikes evolution per day
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.date,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```

    /anaconda3/lib/python3.7/site-packages/scipy/stats/stats.py:1713: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval



![png](output_34_1.png)


Based on the two years dataset, it seems that the utilization of the bike sharing service has increased over the period. The number of bikes rented per day also seems to vary depending on the season, with Spring and Summer months being showing a higher utilization of the service.

#### Bike sharing utilization by Month


```python
# Total_bikes by Month - Line Plot
plt.figure(figsize=(15,5))
g = sns.lineplot(x = hours_df.month,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12])
plt.tight_layout()
```


![png](output_37_0.png)



```python
# Total_bikes by Month - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.month,
            y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.tight_layout()
```


![png](output_38_0.png)


The average utilization per month seems to increase between April and October, with a higher variance too.

#### Bike sharing utilization by Hour


```python
# Total_bikes by Hour - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.hour,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.xticks([0, 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
plt.tight_layout()
```


![png](output_41_0.png)



```python
# Total_bikes by Hour - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.hour,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_42_0.png)



```python
# Total_bikes by Hour - Distribution
plt.figure(figsize=(15,5))
sns.distplot(hours_df.total_bikes,
             bins = 100,
             color = 'steelblue').axes.set(xlim = (min(hours_df.total_bikes),max(hours_df.total_bikes)),
                                           xticks = [0,100,200,300,400,500,600,700,800,900,1000])
plt.tight_layout()
```


![png](output_43_0.png)


The utilization seems really similar over the day, with 2 peaks around 8am and between 5pm and 6pm. The box plot shows potential outliers in the data, which will be removed after the Feature Construction stage. It also highlight an important variance during day time, especially at peak times. The distribution plot shows that utilization is most of the time below 40 bikes simultaneously, and can reach about 1,000 bikes.

#### Bike sharing utilization by Season


```python
# Total_bikes by Season - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.season,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'])
plt.xticks([1,2,3,4])
plt.tight_layout()
```


![png](output_46_0.png)



```python
# Total_bikes by Season - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.season,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Winter', 'Spring', 'Summer', 'Fall'])
plt.tight_layout()
```


![png](output_47_0.png)


Summer appears to be the high season, with Spring and Fall having similar utilization shapes. Winter logically appears to be the low season with, however, potential utilization peaks which can reach the same number of bikes than in high season.

#### Bike sharing utilization by Holiday


```python
# Total_bikes by Holidays - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.holiday,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Normal Day', 'Holiday'])
plt.tight_layout()
```


![png](output_50_0.png)


Utilization of bikes during holidays seems lower and with less peaks.

#### Bike sharing utilization by Weekday


```python
# Total_bikes by Weekday - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.weekday,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.xticks([0,1,2,3,4,5,6])
plt.tight_layout()
```


![png](output_53_0.png)



```python
# Total_bikes by Weekday - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.weekday,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])
plt.tight_layout()
```


![png](output_54_0.png)


The average utilization per hour seems higher at the end of the week, but overall, weekends appear to have lower frequentation and weekdays have higher peaks.

#### Bike sharing utilization by Working Day


```python
# Total_bikes by Working Day - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.workingday,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Non Working Day', 'Working Day'])
plt.tight_layout()
```


![png](output_57_0.png)


Utilization seems higher during working days, with higher peaks.

#### Bike sharing utilization by Weather Condition


```python
# Total_bikes by Weather Condition - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.weather_condition,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Clear', 'Mist', 'Light Rain', 'Heavy Rain'])
plt.xticks([1,2,3,4])
plt.tight_layout()
```


![png](output_60_0.png)



```python
# Total_bikes by Weather Condition - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.weather_condition,
             y = hours_df.total_bikes,
             color = 'steelblue') \
   .axes.set_xticklabels(['Clear', 'Mist', 'Light Rain', 'Heavy Rain'])
plt.tight_layout()
```


![png](output_61_0.png)


Unsurprisingly, bike sharing utilization is getting worse with bad weather.

#### Bike sharing utilization by Actual Temperature


```python
# Total_bikes by Actual Temperature - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.actual_temp,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_64_0.png)



```python
# Total_bikes by Actual Temperature - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.actual_temp,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_65_0.png)


The utilization is almost inexistant for sub-zero temperatures. It then grows with the increase of temperature, but drops down when it gets extremely hot.

#### Bike sharing utilization by Feeling Temperature


```python
# Total_bikes by Feeling Temperature - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.feeling_temp,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_68_0.png)



```python
# Total_bikes by Feeling Temperature - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.feeling_temp,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_69_0.png)


The utilization by feeling temperature follows the same rules than by actual temperature.

#### Bike sharing utilization by Humidity


```python
# Total_bikes by Humidity - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.humidity,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_72_0.png)



```python
# Total_bikes by Humidity - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.humidity,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_73_0.png)


The utilization of bike sharing services is decreasing with the increase of humidity.

#### Bike sharing utilization by Wind Speed


```python
# Total_bikes by Wind Speed - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = hours_df.windspeed,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_76_0.png)



```python
# Total_bikes by Wind Speed - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.windspeed,
             y = hours_df.total_bikes,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_77_0.png)


Stronger wind seems to discourage users to use the bike sharing service.

#### Bike sharing utilization by Casual


```python
# Total_bikes by Casual - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(y = hours_df.casual,
             x = hours_df.total_bikes,
             color = 'steelblue')
sns.lineplot(y = hours_df.total_bikes,
             x = hours_df.total_bikes,
             color = 'orange')
plt.tight_layout()
```


![png](output_80_0.png)



```python
# Total_bikes by Casual - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(y = hours_df.casual,
            x = hours_df.total_bikes,
             color = 'steelblue')
sns.lineplot(y = hours_df.total_bikes,
             x = hours_df.total_bikes,
             color = 'orange')
plt.tight_layout()
```


![png](output_81_0.png)


The number of casual users seems to be quite low compared to the total users, but there are peaks of activity when total utilization reaches values between 500 and 800 bikes.

#### Bike sharing utilization by Registered


```python
# Total_bikes by Registered - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(y = hours_df.registered,
             x = hours_df.total_bikes,
             color = 'steelblue')
sns.lineplot(y = hours_df.total_bikes,
             x = hours_df.total_bikes,
             color = 'orange')
plt.tight_layout()
```


![png](output_84_0.png)



```python
# Total_bikes by Registered - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(y = hours_df.registered,
            x = hours_df.total_bikes,
             color = 'steelblue')
sns.lineplot(y = hours_df.total_bikes,
             x = hours_df.total_bikes,
             color = 'orange')
plt.tight_layout()
```


![png](output_85_0.png)


The number of registered users is usually high, compared to the total number of bikes. There are however drops between 500 and 800 total users.

#### Casual vs Registered Users


```python
cas_reg = pd.DataFrame(hours_df.registered)
cas_reg['casual'] = hours_df.casual
cas_reg['total_bikes'] = hours_df.total_bikes
cas_reg['ratio_cas_tot'] = np.where(cas_reg.total_bikes == 0,0,round(cas_reg.casual / cas_reg.total_bikes,4))
```


```python
# Ratio of Casual Users - Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(y = cas_reg.ratio_cas_tot,
             x = cas_reg.total_bikes,
             color = 'steelblue')
plt.axhline(1, color='orange')
plt.tight_layout()
```


![png](output_89_0.png)



```python
# Ratio of Casual Users - Box Plot
plt.figure(figsize=(15,5))
sns.boxplot(y = cas_reg.ratio_cas_tot,
            x = cas_reg.total_bikes,
             color = 'steelblue')
plt.axhline(1, color='orange')
plt.tight_layout()
```


![png](output_90_0.png)



```python
# Ratio of Casual Users - Distribution
plt.figure(figsize=(15,5))
sns.distplot(cas_reg.ratio_cas_tot,
             bins = 100,
             color = 'steelblue').axes.set(xlim = (min(cas_reg.ratio_cas_tot),max(cas_reg.ratio_cas_tot)))
plt.axvline(0.5, color='orange', linestyle='--')
plt.tight_layout()
```


![png](output_91_0.png)


The ratio of casual users is most of the time lower than the ratio of registered users, mainly lower than 30% of total users.

#### Total_Bikes by Hour with Holiday Hue


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.total_bikes,
             x = hours_df.hour,
             hue = hours_df.holiday.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Normal Day', 'Holiday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_94_0.png)


The utilization by hour during normal days differs from the utilization during holidays. During normal days, two (2) peaks are present during commute times (around 8am and 5-6pm), while during holidays, utilization is higher during the day between 10am and 8pm. Utilization during holidays also shows a higher variance.

#### Total_Bikes by Hour with Working Day Hue


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.total_bikes,
             x = hours_df.hour,
             hue = hours_df.workingday.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Non Working Day', 'Working Day']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.axhline(hours_df.total_bikes.mean()+0.31*hours_df.total_bikes.mean(), color='orange')
plt.tight_layout()
```


![png](output_97_0.png)


Quite similar than the utilization by holiday, the utilization by hour during working days differs from the utilization during non working days. During working days, two (2) peaks are present during commute times (around 8am and 5-6pm), while during non working days, utilization is higher during the day between 10am and 8pm. Interestingly, utilization during non working day seems to have less variance than during holidays.

#### Total_Bikes by Hour with Weekday Hue


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.total_bikes,
             x = hours_df.hour,
             hue = hours_df.weekday.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_100_0.png)


The utilization by hour during weekdays differs from the utilization during weekends. During weekdays, two (2) peaks are present during commute times (around 8am and 5-6pm), while during weekends, utilization is higher during the day between 10am and 6pm.

#### Total_Bikes by Hour with Weekday Hue for Registered Users


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.registered,
             x = hours_df.hour,
             hue = hours_df.weekday.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_103_0.png)


Registered users seem to be responsible for the two (2) peaks during commute times. They still use the bikes during the weekends.

#### Total_Bikes by Hour with Weekday Hue for Casual Users


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.casual,
             x = hours_df.hour,
             hue = hours_df.weekday.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_106_0.png)


Casual users are mainly using the bikes during the weekends.

#### Total_Bikes by Hour with Weather Conditions Hue


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.total_bikes,
             x = hours_df.hour,
             hue = hours_df.weather_condition.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Clear', 'Mist', 'Light Rain', 'Heavy Rain']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_109_0.png)


The weather seems to have a consistent impact on the utilization by hour, except for mist which doesn't seem to discourage morning commuters.

#### Total_Bikes by Hour with Seasons Hue


```python
plt.figure(figsize=(15,5))
g = sns.pointplot(y = hours_df.total_bikes,
             x = hours_df.hour,
             hue = hours_df.season.astype('int'),
             palette = 'viridis',
             markers='.',
             errwidth = 1.5)
g_legend = g.axes.get_legend()
g_labels = ['Winter', 'Spring', 'Summer', 'Fall']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)
plt.tight_layout()
```


![png](output_112_0.png)


The season seems to have a consistent impact on the utilization by hour.

#### Humidity by Month


```python
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.actual_temp,
            y = hours_df.humidity,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_115_0.png)


The season seems to have a consistent impact on the utilization by hour.

#### Correlation Analysis

A correlation analysis will allow to identify relationships between the dataset variables. A plot of their distributions highlighting the value of the target variable might also reveal some patterns.


```python
hours_df_corr = hours_df.copy()
hours_df_corr = hours_df_corr.drop(['date', 'year', 'month', 'hour', 'casual', 'registered', 'total_bikes'], axis=1)
for column in hours_df_corr.columns:
    hours_df_corr[column] = hours_df_corr[column].astype('float')
    
plt.figure(figsize=(12, 10))
sns.heatmap(hours_df_corr.corr(), 
            cmap=sns.diverging_palette(220, 20, n=7), vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);
```


![png](output_119_0.png)



```python
fig = plt.figure(figsize=(15, 10))
axs = fig.subplots(2,4)

sns.scatterplot(hours_df['actual_temp'], hours_df['feeling_temp'], palette=('viridis'), ax = axs[0,0])

sns.scatterplot(hours_df['humidity'],hours_df['windspeed'], palette=('viridis'), ax = axs[0,1])

sns.countplot(hours_df['holiday'],hue=hours_df['workingday'], palette=('viridis'), ax = axs[0,2])
axs[0,2].set_xticklabels(labels=['Normal Day', 'Holiday'])
g_legend = axs[0,2].get_legend()
g_labels = ['Non Working', 'Working']
for t, l in zip(g_legend.texts, g_labels): t.set_text(l)

sns.boxplot(hours_df['weather_condition'], hours_df['humidity'], palette=('viridis'), ax = axs[0,3])
axs[0,3].set_xticklabels(labels=['Clear', 'Mist', 'L. Rain', 'H. Rain'])

sns.boxplot(hours_df['season'], hours_df['actual_temp'], palette=('viridis'), ax = axs[1,0])
axs[1,0].set_xticklabels(labels=['Winter', 'Spring', 'Summer', 'Fall'])

sns.boxplot(hours_df['season'], hours_df['feeling_temp'], palette=('viridis'), ax = axs[1,1])
axs[1,1].set_xticklabels(labels=['Winter', 'Spring', 'Summer', 'Fall'])

sns.boxplot(hours_df['season'], hours_df['humidity'], palette=('viridis'), ax = axs[1,2])
axs[1,2].set_xticklabels(labels=['Winter', 'Spring', 'Summer', 'Fall'])

sns.boxplot(hours_df['season'], hours_df['windspeed'], palette=('viridis'), ax = axs[1,3])
axs[1,3].set_xticklabels(labels=['Winter', 'Spring', 'Summer', 'Fall'])

fig.tight_layout()
```


![png](output_120_0.png)


The correlation matrix shows a high correlation between `actual_temp` and `feeling_temp`. Thus, only the `actual_temp` variable will be used in the study, and the `feeling_temp` will be removed from the dataset.

Another interesting relationship exists between `holiday` and `workingday`. Every holiday is a non-working day. Based on previous plots, the utilization of bikes per hour based on `workingday` seems to be more stable than based on `holiday`, thus the variable `holiday` will be removed.

Some other logical correlations can be found between meteorological conditions and seasons, but they are not strong enough to lighten the dataset.

### Scaling and Skewness


```python
hours_prep_scaled = hours_df.copy().drop(['date','casual', 'registered', 'holiday','feeling_temp'],axis=1)
```


```python
hours_prep_scaled.describe()
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
      <th>actual_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.496987</td>
      <td>0.627229</td>
      <td>0.190098</td>
      <td>189.463088</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.192556</td>
      <td>0.192930</td>
      <td>0.122340</td>
      <td>181.387599</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.340000</td>
      <td>0.480000</td>
      <td>0.104500</td>
      <td>40.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.500000</td>
      <td>0.630000</td>
      <td>0.194000</td>
      <td>142.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.660000</td>
      <td>0.780000</td>
      <td>0.253700</td>
      <td>281.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.850700</td>
      <td>977.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
scaler = MinMaxScaler()
hours_prep_scaled[['actual_temp', 'humidity', 'windspeed', 'total_bikes']] = pd.DataFrame(scaler.fit_transform(hours_prep_scaled[['actual_temp', 'humidity','windspeed', 'total_bikes']]))
hours_prep_scaled.describe()
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:323: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)





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
      <th>actual_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>total_bikes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
      <td>17379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.486722</td>
      <td>0.627229</td>
      <td>0.223460</td>
      <td>0.193097</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.196486</td>
      <td>0.192930</td>
      <td>0.143811</td>
      <td>0.185848</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.326531</td>
      <td>0.480000</td>
      <td>0.122840</td>
      <td>0.039959</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.489796</td>
      <td>0.630000</td>
      <td>0.228047</td>
      <td>0.144467</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.653061</td>
      <td>0.780000</td>
      <td>0.298225</td>
      <td>0.286885</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
def feature_skewness(df):
    numeric_dtypes = ['int16', 'int32', 'int64', 
                      'float16', 'float32', 'float64']
    numeric_features = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes: 
            numeric_features.append(i)

    feature_skew = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':feature_skew})
    return feature_skew, numeric_features

def fix_skewness(df):
    feature_skew, numeric_features = feature_skewness(df)
    high_skew = feature_skew[feature_skew > 0.5]
    skew_index = high_skew.index
    
    for i in skew_index:
        df[i] = boxcox1p(df[i], boxcox_normmax(df[i]+1))

    skew_features = df[numeric_features].apply(
        lambda x: skew(x)).sort_values(ascending=False)
    skews = pd.DataFrame({'skew':skew_features})
    return df
```


```python
hours_df_num = hours_df.select_dtypes(include = ['float64', 'int64']);
hours_prep_scaled.hist(figsize=(15, 5), bins=50, xlabelsize=3, ylabelsize=3, color='steelblue');
```


![png](output_127_0.png)



```python
hours_prep_skew = fix_skewness(hours_prep_scaled)
hours_prep_skew.hist(figsize=(15, 5), bins=50, xlabelsize=3, ylabelsize=3, color='steelblue');
```


![png](output_128_0.png)


The scale and skewness of the dataset are corrected.

### Encoding Categorical Variables


```python
hours_prep_encoded = hours_prep_skew.copy()
```


```python
def date_features(df):
    columns = df.columns
    return df.select_dtypes(include=[np.datetime64]).columns

def numerical_features(df):
    columns = df.columns
    return df._get_numeric_data().columns

def categorical_features(df):
    numerical_columns = numerical_features(df)
    date_columns = date_features(df)
    return(list(set(df.columns) - set(numerical_columns) - set(date_columns) ))

def onehot_encode(df):
    numericals = df.get(numerical_features(df))
    new_df = numericals.copy()
    for categorical_column in categorical_features(df):
        new_df = pd.concat([new_df, 
                            pd.get_dummies(df[categorical_column], 
                                           prefix=categorical_column)], 
                           axis=1)
    return new_df

def onehot_encode_single(df, col_to_encode, drop = True):
    if type(col_to_encode) != str:
        raise TypeError ('col_to_encode should be a string.')
    new_df = df.copy()
    
    if drop == True:
        new_df = new_df.drop([col_to_encode], axis=1)

    new_df = pd.concat([new_df, 
                        pd.get_dummies(df[col_to_encode],
                                       prefix=col_to_encode)],
                       axis=1)
    return new_df
```


```python
hours_clean = onehot_encode(hours_prep_encoded)
df_desc(hours_clean)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_7</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_8</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_9</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_13</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_14</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_16</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_17</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_18</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_19</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_20</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_21</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_22</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_23</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_7</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_8</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_9</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename columns
hours_clean.rename(columns={'year_0':'year_2011',
                        'year_1':'year_2012',
                        'season_1':'season_winter',
                        'season_2':'season_spring',
                        'season_3':'season_summer',
                        'season_4':'season_fall',
                        'workingday_0':'workingday_no',
                        'workingday_1':'workingday_yes',
                        'month_1':'month_jan',
                        'month_2':'month_feb',
                        'month_3':'month_mar',
                        'month_4':'month_apr',
                        'month_5':'month_may',
                        'month_6':'month_jun',
                        'month_7':'month_jul',
                        'month_8':'month_aug',
                        'month_9':'month_sep',
                        'month_10':'month_oct',
                        'month_11':'month_nov',
                        'month_12':'month_dec',
                        'weather_condition_1':'weather_condition_clear',
                        'weather_condition_2':'weather_condition_mist',
                        'weather_condition_3':'weather_condition_light_rain',
                        'weather_condition_4':'weather_condition_heavy_rain',
                        'weekday_0':'weekday_sunday',
                        'weekday_1':'weekday_monday',
                        'weekday_2':'weekday_tuesday',
                        'weekday_3':'weekday_wednesday',
                        'weekday_4':'weekday_thursday',
                        'weekday_5':'weekday_friday',
                        'weekday_6':'weekday_saturday'},
                inplace=True)
```


```python
hours_clean.drop('workingday_no', inplace = True, axis = 1)
```

The variable `workingday_no` has been removed, as complementary of `workingday_yes`.


```python
hours_clean.head()
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
      <th>actual_temp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>total_bikes</th>
      <th>weekday_sunday</th>
      <th>weekday_monday</th>
      <th>weekday_tuesday</th>
      <th>weekday_wednesday</th>
      <th>weekday_thursday</th>
      <th>weekday_friday</th>
      <th>weekday_saturday</th>
      <th>hour_0</th>
      <th>hour_1</th>
      <th>hour_2</th>
      <th>hour_3</th>
      <th>hour_4</th>
      <th>hour_5</th>
      <th>hour_6</th>
      <th>hour_7</th>
      <th>hour_8</th>
      <th>hour_9</th>
      <th>hour_10</th>
      <th>hour_11</th>
      <th>hour_12</th>
      <th>hour_13</th>
      <th>hour_14</th>
      <th>hour_15</th>
      <th>hour_16</th>
      <th>hour_17</th>
      <th>hour_18</th>
      <th>hour_19</th>
      <th>hour_20</th>
      <th>hour_21</th>
      <th>hour_22</th>
      <th>hour_23</th>
      <th>workingday_yes</th>
      <th>year_2011</th>
      <th>year_2012</th>
      <th>season_winter</th>
      <th>season_spring</th>
      <th>season_summer</th>
      <th>season_fall</th>
      <th>weather_condition_clear</th>
      <th>weather_condition_mist</th>
      <th>weather_condition_light_rain</th>
      <th>weather_condition_heavy_rain</th>
      <th>month_jan</th>
      <th>month_feb</th>
      <th>month_mar</th>
      <th>month_apr</th>
      <th>month_may</th>
      <th>month_jun</th>
      <th>month_jul</th>
      <th>month_aug</th>
      <th>month_sep</th>
      <th>month_oct</th>
      <th>month_nov</th>
      <th>month_dec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.224490</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>0.014911</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.204082</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>0.036985</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.204082</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>0.029859</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.224490</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0.012001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.224490</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The dataset is ready for modeling.

### Training/Test Split


```python
def list_features(df, target):
    features = list(df)
    features.remove(target)
    return features
```


```python
target = 'total_bikes'
features = list_features(hours_clean, target)
```


```python
X = hours_clean[features]
X_train = X.loc[(X['year_2011']==1) | ((X['year_2012']==1) & (X['month_sep']==0) & (X['month_oct']==0) & (X['month_nov']==0) & (X['month_dec']==0)),features]
X_test = X.loc[(X['year_2012']==1) & ((X['month_sep']==1) | (X['month_oct']==1) | (X['month_nov']==1) | (X['month_dec']==1)),features]
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('X_train:', X_train.shape[0],'rows |', X_train.shape[1], 'columns'))
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('X_test:', X_test.shape[0],'rows |', X_test.shape[1], 'columns'))
```

    X_train:   14491 rows |  57 columns
    X_test:     2888 rows |  57 columns



```python
X_train.groupby(['year_2011','year_2012','month_jan','month_feb','month_mar','month_apr','month_may','month_jun','month_jul','month_aug','month_sep','month_oct','month_nov','month_dec']).size().reset_index()
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
      <th>year_2011</th>
      <th>year_2012</th>
      <th>month_jan</th>
      <th>month_feb</th>
      <th>month_mar</th>
      <th>month_apr</th>
      <th>month_may</th>
      <th>month_jun</th>
      <th>month_jul</th>
      <th>month_aug</th>
      <th>month_sep</th>
      <th>month_oct</th>
      <th>month_nov</th>
      <th>month_dec</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>718</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>743</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>692</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>741</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>741</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>719</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>743</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>717</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>731</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>719</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>730</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>649</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>688</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test.groupby(['year_2011','year_2012','month_jan','month_feb','month_mar','month_apr','month_may','month_jun','month_jul','month_aug','month_sep','month_oct','month_nov','month_dec']).size().reset_index()
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
      <th>year_2011</th>
      <th>year_2012</th>
      <th>month_jan</th>
      <th>month_feb</th>
      <th>month_mar</th>
      <th>month_apr</th>
      <th>month_may</th>
      <th>month_jun</th>
      <th>month_jul</th>
      <th>month_aug</th>
      <th>month_sep</th>
      <th>month_oct</th>
      <th>month_nov</th>
      <th>month_dec</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>742</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>718</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>708</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>720</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = hours_clean.copy()
y_train = y.loc[(y['year_2011']==1) | ((y['year_2012']==1) & (y['month_sep']==0) & (y['month_oct']==0) & (y['month_nov']==0) & (y['month_dec']==0)),:]
y_test = y.loc[(y['year_2012']==1) & ((y['month_sep']==1) | (y['month_oct']==1) | (y['month_nov']==1) | (y['month_dec']==1)),:]
y_train = pd.DataFrame(y_train[target])
y_test = pd.DataFrame(y_test[target])
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('y_train:', y_train.shape[0],'rows |', y_train.shape[1], 'columns'))
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('y_test:', y_test.shape[0],'rows |', y_test.shape[1], 'columns'))
```

    y_train:   14491 rows |   1 columns
    y_test:     2888 rows |   1 columns



```python
print('{:<35} {!r:>}'.format('Same indexes for X_train and y_train:', X_train.index.values.tolist() == y_train.index.values.tolist()))
print('{:<35} {!r:>}'.format('Same indexes for X_test and y_test:', X_test.index.values.tolist() == y_test.index.values.tolist()))
```

    Same indexes for X_train and y_train: True
    Same indexes for X_test and y_test: True



```python
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features:',X.shape[0], 'items | ', X.shape[1],'columns'))
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features Train:',X_train.shape[0], 'items | ', X_train.shape[1],'columns'))
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features Test:',X_test.shape[0], 'items | ',  X_test.shape[1],'columns'))
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target:',y.shape[0], 'items | ', 1,'columns'))
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target Train:',y_train.shape[0], 'items | ', 1,'columns'))
print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target Test:',y_test.shape[0], 'items | ', 1,'columns'))
```

    Features:        17379 items |  57 columns
    Features Train:  14491 items |  57 columns
    Features Test:    2888 items |  57 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns


The Train Set is arbitrarily defined as all records until August 31st 2012, and the Test Set all records from September 1st 2012. Below function will be used to repeat the operation on future dataframes including new features.


```python
def train_test_split_0(df, target, features):
    X = df[features]
    y = pd.DataFrame(df[target])
    X_train = X.loc[(X['year_2011']==1) | ((X['year_2012']==1) & (X['month_sep']==0) & (X['month_oct']==0) & (X['month_nov']==0) & (X['month_dec']==0)),features]
    X_test = X.loc[(X['year_2012']==1) & ((X['month_sep']==1) | (X['month_oct']==1) | (X['month_nov']==1) | (X['month_dec']==1)),features]
    y_train = y.iloc[X_train.index.values.tolist()]
    y_test = y.iloc[X_test.index.values.tolist()]
    
    print('{:<40} {!r:>}'.format('Same indexes for X and y:', X.index.values.tolist() == y.index.values.tolist()))
    print('{:<40} {!r:>}'.format('Same indexes for X_train and y_train:', X_train.index.values.tolist() == y_train.index.values.tolist()))
    print('{:<40} {!r:>}'.format('Same indexes for X_test and y_test:', X_test.index.values.tolist() == y_test.index.values.tolist()))
    print()
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features:',X.shape[0], 'items | ', X.shape[1],'columns'))
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features Train:',X_train.shape[0], 'items | ', X_train.shape[1],'columns'))
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Features Test:',X_test.shape[0], 'items | ',  X_test.shape[1],'columns'))
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target:',y.shape[0], 'items | ', 1,'columns'))
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target Train:',y_train.shape[0], 'items | ', 1,'columns'))
    print('{:<15} {:>6} {:>6} {:>2} {:>6}'.format('Target Test:',y_test.shape[0], 'items | ', 1,'columns'))
    print()
    
    return X, X_train, X_test, y, y_train, y_test
```

## Baseline


```python
lm = linear_model.LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)

print('Intercept:', lm.intercept_)
print('Coefficients:', lm.coef_)
print('Mean squared error (MSE): {:.2f}'.format(mean_squared_error(y_test, y_pred)))
print('Variance score (R2): {:.2f}'.format(r2_score(y_test, y_pred)))
```

    Intercept: [9.17949121e+11]
    Coefficients: [[ 9.69846750e-02 -2.98091413e-02 -1.88745142e-02 -1.36950003e+12
      -1.36950003e+12 -1.36950003e+12 -1.36950003e+12 -1.36950003e+12
      -1.36950003e+12 -1.36950003e+12 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11  8.30078125e-03 -1.66365970e+11
      -1.66365970e+11  3.63115327e+11  3.63115327e+11  3.63115327e+11
       3.63115327e+11  2.35643826e+11  2.35643826e+11  2.35643826e+11
       2.35643826e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.76


The baseline model for our dataset obtains a R square of 0.76 with 57 features.

## Feature Engineering

### Cross Validation Strategy


```python
def cross_val_ts(algorithm, X_train, y_train, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = cross_validate(algorithm, X_train, y_train, cv=tscv,
                            scoring=('r2'),
                            return_train_score=True)
    print('Cross Validation Variance score (R2): {:.2f}'.format(scores['train_score'].mean()))
```


```python
cross_val_ts(lm,X_train, y_train, 10)
```

    Cross Validation Variance score (R2): 0.75


The cross validation used is a recursive time series split with 10 folds.

### Features Construction

#### Pipeline

Each new feature will be tested through below pipeline.


```python
def pipeline(df, target, algorithm, n_splits = 10):
    features = list_features(df, target)
    X, X_train, X_test, y, y_train, y_test = train_test_split_0(df, target, features)
    cross_val_ts(algorithm,X_train, y_train, n_splits)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)

    print()
    print('Intercept:', lm.intercept_)
    print('Coefficients:', lm.coef_)
    print('Mean squared error (MSE): {:.2f}'.format(mean_squared_error(y_test, y_pred)))
    print('Variance score (R2): {:.2f}'.format(r2_score(y_test, y_pred)))
```


```python
pipeline(hours_clean, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  57 columns
    Features Train:  14491 items |  57 columns
    Features Test:    2888 items |  57 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.75
    
    Intercept: [9.17949121e+11]
    Coefficients: [[ 9.69846750e-02 -2.98091413e-02 -1.88745142e-02 -1.36950003e+12
      -1.36950003e+12 -1.36950003e+12 -1.36950003e+12 -1.36950003e+12
      -1.36950003e+12 -1.36950003e+12 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11 -3.14853433e+11 -3.14853433e+11
      -3.14853433e+11 -3.14853433e+11  8.30078125e-03 -1.66365970e+11
      -1.66365970e+11  3.63115327e+11  3.63115327e+11  3.63115327e+11
       3.63115327e+11  2.35643826e+11  2.35643826e+11  2.35643826e+11
       2.35643826e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11  3.34011164e+11  3.34011164e+11  3.34011164e+11
       3.34011164e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.76



```python
hours_FE_sel = hours_clean.copy()
```

#### Day

The variable `day` is added to understand if patterns exist based on specific dates.


```python
# Add the day from 'date'
hours_FE1 = pd.concat([hours_FE_sel,pd.DataFrame(pd.DatetimeIndex(hours_df['date']).day)], axis=1, sort=False, ignore_index=False)
hours_FE1.rename(columns={'date':'day'}, inplace=True)

# Encode feature
hours_FE1 = onehot_encode_single(hours_FE1, 'day')
df_desc(hours_FE1)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_sunday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_monday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_tuesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_wednesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_thursday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_friday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_saturday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_7</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_8</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_9</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_13</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_14</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_16</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_17</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_18</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <th>day_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_7</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_8</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_9</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_13</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_14</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_16</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_17</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_18</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_19</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_20</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_21</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_22</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_23</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_24</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_25</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_26</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_27</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_28</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_29</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_30</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>day_31</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>89 rows × 6 columns</p>
</div>




```python
pipeline(hours_FE1, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  88 columns
    Features Train:  14491 items |  88 columns
    Features Test:    2888 items |  88 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.76
    
    Intercept: [2.61727993e+11]
    Coefficients: [[ 9.83215701e-02 -2.93104942e-02 -1.92630983e-02 -1.49087851e+11
      -1.49087851e+11 -1.49087851e+11 -1.49087851e+11 -1.49087851e+11
      -1.49087851e+11 -1.49087851e+11  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  8.17461419e+10  8.17461419e+10
       8.17461419e+10  8.17461419e+10  1.11083984e-02  7.86603338e+10
       7.86603338e+10 -1.95755300e+11 -1.95755300e+11 -1.95755300e+11
      -1.95755300e+11 -3.62162271e+11 -3.62162271e+11 -3.62162271e+11
      -3.62162271e+11  3.13414481e+11  3.13414481e+11  3.13414481e+11
       3.13414481e+11  3.13414481e+11  3.13414481e+11  3.13414481e+11
       3.13414481e+11  3.13414481e+11  3.13414481e+11  3.13414481e+11
       3.13414481e+11 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10
      -2.85435265e+10 -2.85435265e+10 -2.85435265e+10 -2.85435265e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.76


This additional feature gives the same result as the baseline, with a cross validation mean of 0.75 (0.75 for the baseline) and a metric of 0.76 (0.76 for the baseline). With the number of added variables and the lack of score improvement, we reject this feature.

#### Month-Day

The variable `month_day` is added to understand if patterns exist based on specific dates.


```python
# Add month-day from 'date'
hours_FE2 = pd.concat([hours_FE_sel,pd.DataFrame(pd.DatetimeIndex(hours_df['date']).strftime('%m-%d'))], axis=1, sort=False, ignore_index=False)
hours_FE2.rename(columns={0:'month_day'}, inplace=True)

# Encode feature
hours_FE2 = onehot_encode_single(hours_FE2, 'month_day')
df_desc(hours_FE2)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_sunday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_monday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_tuesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_wednesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_thursday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_friday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_saturday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_0</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_1</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_2</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_3</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_4</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_5</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_6</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_7</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_8</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_9</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_13</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_14</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_16</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_17</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hour_18</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
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
      <th>month_day_12-02</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-03</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-04</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-05</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-06</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-07</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-08</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-09</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-10</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-12</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-13</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-14</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-16</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-17</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-18</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-19</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-20</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-21</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-22</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-23</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-24</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-25</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-26</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-27</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-28</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-29</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-30</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_day_12-31</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>424 rows × 6 columns</p>
</div>




```python
pipeline(hours_FE2, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  423 columns
    Features Train:  14491 items |  423 columns
    Features Test:    2888 items |  423 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.78
    
    Intercept: [-4.13022056e+11]
    Coefficients: [[ 1.09935589e-01 -2.97102286e-02 -6.43684294e-03 -8.10099124e+10
      -8.10099124e+10 -8.10099124e+10 -8.10099124e+10 -8.10099124e+10
      -8.10099124e+10 -8.10099124e+10 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11 -1.55736820e+11 -1.55736820e+11
      -1.55736820e+11 -1.55736820e+11  9.76562500e-04  1.47645299e+11
       1.47645299e+11  1.89624714e+11  7.77464299e+10  3.66059174e+11
       3.60960627e+11 -5.02418817e+09 -5.02418817e+09 -5.02418817e+09
      -5.02418817e+09  2.38373069e+11  3.55092383e+11  3.67080336e+11
       5.53240274e+11  5.92095627e+11  1.98397050e+11  1.43873969e+11
       2.22287028e+10  2.69262803e+11 -6.01395036e+09  2.07875731e+11
       2.33180414e+11  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
       7.91498937e+10  7.91498937e+10  7.91498937e+10  7.91498937e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -3.75694200e+10 -3.75694200e+10 -3.75694200e+10
      -3.75694200e+10 -4.95573734e+10 -4.95573734e+10 -4.95573734e+10
      -4.95573734e+10 -4.95573734e+10 -4.95573734e+10 -4.95573734e+10
      -4.95573734e+10 -4.95573734e+10 -4.95573734e+10 -4.95573734e+10
      -4.95573734e+10 -4.95573734e+10 -4.95573734e+10 -4.95573734e+10
      -4.95573734e+10 -4.95573734e+10 -4.95573734e+10 -4.95573734e+10
      -4.95573734e+10  6.23209108e+10  6.23209108e+10  6.23209108e+10
       6.23209108e+10  6.23209108e+10  6.23209108e+10  6.23209108e+10
       6.23209108e+10  6.23209108e+10  6.23209108e+10  6.23209108e+10
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.23839026e+11 -1.23839026e+11
      -1.23839026e+11 -1.23839026e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11 -1.62694380e+11 -1.62694380e+11 -1.62694380e+11
      -1.62694380e+11  2.31004197e+11  2.31004197e+11  2.31004197e+11
       2.31004197e+11  2.31004197e+11  2.31004197e+11  2.31004197e+11
       2.31004197e+11  2.31004197e+11  2.31004197e+11  2.31004197e+11
       2.31004197e+11  2.31004197e+11  2.31004197e+11  2.31004197e+11
       2.31004197e+11  2.31004197e+11  2.31004197e+11  2.31004197e+11
       2.31004197e+11 -5.73085471e+10 -5.73085471e+10 -5.73085471e+10
      -5.73085471e+10 -5.73085471e+10 -5.73085471e+10 -5.73085471e+10
      -5.73085471e+10 -5.73085471e+10 -5.73085471e+10 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09 -2.78546643e+09 -2.78546643e+09
      -2.78546643e+09 -2.78546643e+09  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11  1.18859800e+11  1.18859800e+11  1.18859800e+11
       1.18859800e+11 -1.28174300e+11 -1.28174300e+11 -1.28174300e+11
      -1.28174300e+11 -1.28174300e+11 -1.28174300e+11 -1.28174300e+11
      -1.28174300e+11 -1.28174300e+11 -1.28174300e+11 -1.28174300e+11
      -1.28174300e+11 -1.28174300e+11 -1.28174300e+11 -1.28174300e+11
      -1.28174300e+11 -1.28174300e+11 -1.28174300e+11 -1.28174300e+11
      -1.28174300e+11 -1.28174300e+11 -1.28174300e+11 -1.23075753e+11
      -1.23075753e+11 -1.23075753e+11 -1.23075753e+11 -1.23075753e+11
      -1.23075753e+11 -1.23075753e+11 -1.23075753e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11  1.52201000e+11  1.52201000e+11
       1.52201000e+11  1.52201000e+11 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -6.16886808e+10 -6.16886808e+10 -6.16886808e+10 -6.16886808e+10
      -8.69933645e+10 -8.69933645e+10 -8.69933645e+10 -8.69933645e+10
      -8.69933645e+10 -8.69933645e+10 -8.69933645e+10 -8.69933645e+10
      -8.69933645e+10 -8.69933645e+10 -8.69933645e+10 -8.69933645e+10
      -8.69933645e+10 -8.69933645e+10 -8.69933645e+10 -8.69933645e+10
      -8.69933645e+10 -8.69933645e+10 -8.69933645e+10 -8.69933645e+10
       8.43425489e+10  8.43425489e+10  8.43425489e+10  8.43425489e+10
       8.43425489e+10  8.43425489e+10  8.43425489e+10  8.43425489e+10
       8.43425489e+10  8.43425489e+10  8.43425489e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.75


This additional feature gives a similar result than the baseline, with a cross validation mean of 0.78 (0.75 for the baseline) and a metric of 0.75 (0.76 for the baseline). With the number of added variables and the lack of score improvement, we reject this feature.

#### Predefined Peaks

The variable `predefined_peak` is added to flag the periods of typically high utilization.


```python
hours_FE3 = hours_FE_sel.copy()

def predefined_peaks(row):
    if (row['workingday_yes'] == 1) & ((row['hour_8'] == 1) |
                                      (row['hour_7'] == 1) |
                                      (row['hour_16'] == 1) |
                                      (row['hour_17'] == 1) |
                                      (row['hour_18'] == 1) |
                                      (row['hour_19'] == 1)):
        return 1
    elif (row['workingday_yes'] == 0) & ((row['hour_11'] == 1) |
                                      (row['hour_10'] == 1) |
                                      (row['hour_12'] == 1) |
                                      (row['hour_13'] == 1) |
                                      (row['hour_14'] == 1) |
                                      (row['hour_15'] == 1) |
                                      (row['hour_16'] == 1) |
                                      (row['hour_17'] == 1) |
                                      (row['hour_19'] == 1) |
                                      (row['hour_18'] == 1)):
        return 1
    else:
        return 0

hours_FE3['predefined_peak'] = hours_FE3.apply(lambda row: predefined_peaks(row), axis=1)
hours_FE3.predefined_peak.value_counts()
```




    0    12085
    1     5294
    Name: predefined_peak, dtype: int64




```python
pipeline(hours_FE3, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  58 columns
    Features Train:  14491 items |  58 columns
    Features Test:    2888 items |  58 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.84
    
    Intercept: [1.48297049e+11]
    Coefficients: [[ 9.60133980e-02 -3.16847930e-02 -2.07489792e-02 -4.95214221e+09
      -4.95214221e+09 -4.95214221e+09 -4.95214221e+09 -4.95214221e+09
      -4.95214221e+09 -4.95214221e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.03931090e+09  2.03931090e+09
       2.03931090e+09  2.03931090e+09  2.38480568e-02 -4.72764005e+10
      -4.72764005e+10 -8.71418052e+10 -8.71418052e+10 -8.71418052e+10
      -8.71418052e+10 -3.17724236e+09 -3.17724236e+09 -3.17724236e+09
      -3.17724236e+09 -7.78876989e+09 -7.78876989e+09 -7.78876989e+09
      -7.78876989e+09 -7.78876989e+09 -7.78876989e+09 -7.78876989e+09
      -7.78876989e+09 -7.78876989e+09 -7.78876989e+09 -7.78876989e+09
      -7.78876989e+09  8.57324600e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.83


This additional feature significantly improves the score of the model, with a cross validation mean of 0.84 (0.75 for the baseline) and a metric of 0.83 (0.76 for the baseline). As this feature also tries to combine hours variables and `workingday_yes`, we assume it might reduce the number of features for the final metric. Thus, we accept this feature.


```python
hours_FE_sel  = hours_FE3.copy()
```

#### Calculated Peaks

The variable `calculated_peak` is added to flag the periods of typically high utilization.


```python
hours_FE4 = hours_FE_sel.copy()

#Set threshold to mean(Total_Bikes)+th based on Trial and Error appraoch
th = 0.315

def calculated_peaks(row, th):
    if (row['total_bikes'] > (hours_FE4.total_bikes.mean()+ th *hours_FE4.total_bikes.mean())):
        return 1
    else:
        return 0

hours_FE4['calculated_peak'] = hours_FE4.apply(lambda row: calculated_peaks(row, th), axis=1)
hours_FE4.calculated_peak.value_counts()
```




    0    11223
    1     6156
    Name: calculated_peak, dtype: int64




```python
pipeline(hours_FE4, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  59 columns
    Features Train:  14491 items |  59 columns
    Features Test:    2888 items |  59 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [-5.35051488e+10]
    Coefficients: [[ 6.52898889e-02 -1.92426421e-02 -1.11274385e-02  9.53347837e+10
       9.53347837e+10  9.53347837e+10  9.53347837e+10  9.53347837e+10
       9.53347837e+10  9.53347837e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10  1.71203613e-02 -8.77850532e+09
      -8.77850532e+09  1.48801267e+10  1.48801267e+10  1.48801267e+10
       1.48801267e+10 -5.02559899e+10 -5.02559899e+10 -5.02559899e+10
      -5.02559899e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  5.19561768e-02  5.74054718e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature significantly improves the score of the model, with a cross validation mean of 0.89 (0.75 for the baseline) and a metric of 0.90 (0.76 for the baseline). Thus, we accept this feature.


```python
hours_FE_sel  = hours_FE4.copy()
```

#### Difference with Season Average Temperature

The variable `diff_season_avg_temp` is added to identify days with clement temperature compared to the seasonal average.


```python
hours_FE5 = hours_FE_sel.copy()

def diff_season_avg_temp_calc(df, row):
    if (row['season_spring'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_spring == 1].mean()
    
    elif (row['season_winter'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_winter == 1].mean()
    
    elif (row['season_fall'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_fall == 1].mean()
    
    elif (row['season_summer'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_summer == 1].mean()
    
hours_FE5['diff_season_avg_temp'] = hours_FE5.apply(lambda row: diff_season_avg_temp_calc(hours_FE5, row), axis=1)
```


```python
pipeline(hours_FE5, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  60 columns
    Features Train:  14491 items |  60 columns
    Features Test:    2888 items |  60 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [1.26255324e+10]
    Coefficients: [[-3.50599669e+07 -1.91853140e-02 -1.10156480e-02 -1.03307983e+10
      -1.03307983e+10 -1.03307983e+10 -1.03307983e+10 -1.03307983e+10
      -1.03307983e+10 -1.03307983e+10  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.07291048e+09  1.07291048e+09
       1.07291048e+09  1.07291048e+09  1.72576904e-02 -5.99564295e+09
      -5.99564295e+09  1.41984599e+10  1.42072434e+10  1.42130299e+10
       1.42028957e+10  2.58096600e+09  2.58096600e+09  2.58096600e+09
       2.58096600e+09 -1.41414409e+10 -1.41414409e+10 -1.41414409e+10
      -1.41414409e+10 -1.41414409e+10 -1.41414409e+10 -1.41414409e+10
      -1.41414409e+10 -1.41414409e+10 -1.41414409e+10 -1.41414409e+10
      -1.41414409e+10  5.20367622e-02  5.73718548e-02  3.50599670e+07]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.89 (0.89 for the previously selected features) and a metric of 0.90 (0.90 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Monthly Average Temperature

The variable `diff_month_avg_temp` is added to identify days with clement temperature compared to the monthly average.


```python
hours_FE6 = hours_FE_sel.copy()

def diff_month_avg_temp_calc(df, row):
    if (row['month_jan'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_jan == 1].mean()
    
    elif (row['month_feb'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_feb == 1].mean()
    
    elif (row['month_mar'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_mar == 1].mean()
    
    elif (row['month_apr'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_apr == 1].mean()
    
    elif (row['month_may'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_may == 1].mean()
    
    elif (row['month_jun'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_jun == 1].mean()
    
    elif (row['month_jul'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_jul == 1].mean()
    
    elif (row['month_aug'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_aug == 1].mean()
    
    elif (row['month_sep'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_sep == 1].mean()
    
    elif (row['month_oct'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_oct == 1].mean()
    
    elif (row['month_nov'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_nov == 1].mean()
    
    elif (row['month_dec'] == 1):
        return row['actual_temp'] - df.actual_temp[df.month_dec == 1].mean()
    
hours_FE6['diff_month_avg_temp'] = hours_FE6.apply(lambda row: diff_month_avg_temp_calc(hours_FE6, row), axis=1)
```


```python
pipeline(hours_FE6, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  60 columns
    Features Train:  14491 items |  60 columns
    Features Test:    2888 items |  60 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [4.66197865e+09]
    Coefficients: [[-3.22124550e+07 -1.91881694e-02 -1.10201277e-02 -7.14579473e+09
      -7.14579473e+09 -7.14579473e+09 -7.14579473e+09 -7.14579473e+09
      -7.14579473e+09 -7.14579473e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.06176659e+09  1.06176659e+09
       1.06176659e+09  1.06176659e+09  1.72481537e-02  1.60673988e+09
       1.60673988e+09  4.55991793e+09  4.55991793e+09  4.55991793e+09
       4.55991793e+09  4.89708871e+09  4.89708871e+09  4.89708871e+09
       4.89708871e+09 -9.63454295e+09 -9.63248661e+09 -9.62950551e+09
      -9.62690491e+09 -9.62280349e+09 -9.61986825e+09 -9.61752223e+09
      -9.61905940e+09 -9.62208995e+09 -9.62634085e+09 -9.63021889e+09
      -9.63169910e+09  5.20343781e-02  5.73674440e-02  3.22124551e+07]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.89 (0.89 for the previously selected features) and a metric of 0.90 (0.90 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Season Average Humidity

The variable `diff_season_avg_humi` is added to identify days with clement humidity compared to the seasonal average.


```python
hours_FE7 = hours_FE_sel.copy()

def diff_season_avg_humi_calc(df, row):
    if (row['season_spring'] == 1):
        return row['humidity'] - df.humidity[df.season_spring == 1].mean()
    
    elif (row['season_winter'] == 1):
        return row['humidity'] - df.humidity[df.season_winter == 1].mean()
    
    elif (row['season_fall'] == 1):
        return row['humidity'] - df.humidity[df.season_fall == 1].mean()
    
    elif (row['season_summer'] == 1):
        return row['humidity'] - df.humidity[df.season_summer == 1].mean()
    
hours_FE7['diff_season_avg_humi'] = hours_FE7.apply(lambda row: diff_season_avg_humi_calc(hours_FE7, row), axis=1)
```


```python
pipeline(hours_FE7, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  60 columns
    Features Train:  14491 items |  60 columns
    Features Test:    2888 items |  60 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [-1.24565675e+11]
    Coefficients: [[ 6.52261289e-02  1.14919607e+11 -1.11663702e-02 -4.05966365e+10
      -4.05966365e+10 -4.05966365e+10 -4.05966365e+10 -4.05966365e+10
      -4.05966365e+10 -4.05966365e+10  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  9.15972884e+09  9.15972884e+09
       9.15972884e+09  9.15972884e+09  1.71737671e-02 -1.38279846e+08
      -1.38279846e+08  4.51423083e+10  3.98935185e+10  3.91873077e+10
       3.52849790e+10  4.23336280e+10  4.23336280e+10  4.23336280e+10
       4.23336280e+10  1.85659370e+09  1.85659370e+09  1.85659370e+09
       1.85659370e+09  1.85659370e+09  1.85659370e+09  1.85659370e+09
       1.85659370e+09  1.85659370e+09  1.85659370e+09  1.85659370e+09
       1.85659370e+09  5.20019531e-02  5.73730469e-02 -1.14919607e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.89 (0.89 for the previously selected features) and a metric of 0.90 (0.90 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Monthly Average Humidity

The variable `diff_month_avg_humi` is added to identify days with clement humidity compared to the monthly average.


```python
hours_FE8 = hours_FE_sel.copy()

def diff_month_avg_humi_calc(df, row):
    if (row['month_jan'] == 1):
        return row['humidity'] - df.humidity[df.month_jan == 1].mean()
    
    elif (row['month_feb'] == 1):
        return row['humidity'] - df.humidity[df.month_feb == 1].mean()
    
    elif (row['month_mar'] == 1):
        return row['humidity'] - df.humidity[df.month_mar == 1].mean()
    
    elif (row['month_apr'] == 1):
        return row['humidity'] - df.humidity[df.month_apr == 1].mean()
    
    elif (row['month_may'] == 1):
        return row['humidity'] - df.humidity[df.month_may == 1].mean()
    
    elif (row['month_jun'] == 1):
        return row['humidity'] - df.humidity[df.month_jun == 1].mean()
    
    elif (row['month_jul'] == 1):
        return row['humidity'] - df.humidity[df.month_jul == 1].mean()
    
    elif (row['month_aug'] == 1):
        return row['humidity'] - df.humidity[df.month_aug == 1].mean()
    
    elif (row['month_sep'] == 1):
        return row['humidity'] - df.humidity[df.month_sep == 1].mean()
    
    elif (row['month_oct'] == 1):
        return row['humidity'] - df.humidity[df.month_oct == 1].mean()
    
    elif (row['month_nov'] == 1):
        return row['humidity'] - df.humidity[df.month_nov == 1].mean()
    
    elif (row['month_dec'] == 1):
        return row['humidity'] - df.humidity[df.month_dec == 1].mean()
    
hours_FE8['diff_month_avg_humi'] = hours_FE8.apply(lambda row: diff_month_avg_humi_calc(hours_FE8, row), axis=1)
```


```python
pipeline(hours_FE8, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  60 columns
    Features Train:  14491 items |  60 columns
    Features Test:    2888 items |  60 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [2.22438373e+11]
    Coefficients: [[ 6.51345891e-02 -1.84637704e+11 -1.06890347e-02  1.81295402e+09
       1.81295402e+09  1.81295402e+09  1.81295402e+09  1.81295402e+09
       1.81295402e+09  1.81295402e+09  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  4.26216755e+08  4.26216755e+08
       4.26216755e+08  4.26216755e+08  1.75895691e-02  5.44995525e+10
       5.44995525e+10 -1.15180723e+11 -1.15180723e+11 -1.15180723e+11
      -1.15180723e+11 -3.03455838e+10 -3.03455838e+10 -3.03455838e+10
      -3.03455838e+10 -2.63890760e+10 -2.89222470e+10 -2.48900324e+10
      -2.50627484e+10 -6.44310564e+09 -2.73353747e+10 -2.32602750e+10
      -1.59946383e+10 -1.73569543e+09 -6.34493866e+09 -1.82438737e+10
      -1.06606651e+10  5.20687103e-02  5.73272705e-02  1.84637704e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.89 (0.89 for the previously selected features) and a metric of 0.90 (0.90 for the previously selected features). With no clear score improvement, we reject this feature.

#### Polynomial Features

The polynomial variables `actual_temp_poly_2`, `actual_temp_poly_3`, `humidity_poly_2`, `humidity_poly_3`, `windspeed_poly_2`, `windspeed_poly_3` are added to identify potential polynomial relationship between the target and the corresponding variables.


```python
hours_FE9 = hours_FE_sel.copy()
degree = 3
poly = PolynomialFeatures(degree)
list_var = ['actual_temp', 'humidity', 'windspeed']

def add_poly(df, degree, list_var):
    poly = PolynomialFeatures(degree)
    for var in list_var:
        poly_mat = poly.fit_transform(df[var].values.reshape(-1,1))
        for i in range(2, degree+1):
            df[var + '_poly_' + str(i)] = poly_mat[:,i]    

add_poly(hours_FE9, degree, list_var)
```


```python
pipeline(hours_FE9, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  65 columns
    Features Train:  14491 items |  65 columns
    Features Test:    2888 items |  65 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.90
    
    Intercept: [6.33948727e+10]
    Coefficients: [[-8.03798050e-02  8.56974462e-02  8.35102451e-03  5.37747263e+09
       5.37747263e+09  5.37747263e+09  5.37747263e+09  5.37747263e+09
       5.37747263e+09  5.37747263e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09 -8.99807164e+09 -8.99807164e+09
      -8.99807164e+09 -8.99807164e+09  1.67932510e-02 -1.65782635e+10
      -1.65782635e+10 -2.01745791e+10 -2.01745791e+10 -2.01745791e+10
      -2.01745791e+10 -1.34546450e+10 -1.34546450e+10 -1.34546450e+10
      -1.34546450e+10 -9.56678603e+09 -9.56678603e+09 -9.56678603e+09
      -9.56678603e+09 -9.56678603e+09 -9.56678603e+09 -9.56678603e+09
      -9.56678603e+09 -9.56678603e+09 -9.56678603e+09 -9.56678603e+09
      -9.56678603e+09  5.27520180e-02  5.62806129e-02  4.04531956e-01
      -3.05279732e-01 -1.35239124e-01  4.66301441e-02 -2.99733877e-02
      -6.23785853e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.90 (0.89 for the previously selected features) and a metric of 0.90 (0.90 for the previously selected features). With no clear score improvement, we reject this feature.

#### Hours Bins

The hour variables seem to be important for the model, but they are numerous. In order to reduce the number of variables, they will be binned into similar ranges.


```python
hours_FE10 = hours_FE_sel.copy()

def bin_hours(row):
    if (row['hour_0'] == 1) | (row['hour_1'] == 1) | (row['hour_2'] == 1) | (row['hour_3'] == 1) | (row['hour_4'] == 1) | (row['hour_5'] == 1):
        return '00-05'
    if (row['hour_6'] == 1) | (row['hour_7'] == 1) | (row['hour_8'] == 1) | (row['hour_9'] == 1):
        return '06-09'
    if (row['hour_10'] == 1) | (row['hour_11'] == 1) | (row['hour_12'] == 1) | (row['hour_13'] == 1) | (row['hour_14'] == 1) | (row['hour_15'] == 1):
        return '10-15'
    if (row['hour_16'] == 1) | (row['hour_17'] == 1) | (row['hour_18'] == 1) | (row['hour_19'] == 1) | (row['hour_20'] == 1):
        return '16-20'
    if (row['hour_21'] == 1) | (row['hour_22'] == 1) | (row['hour_23'] == 1):
        return '21-23'

hours_FE10['hours'] = hours_FE10.apply(lambda row: bin_hours(row), axis=1)
hours_FE10.hours.value_counts()
```




    10-15    4369
    00-05    4276
    16-20    3644
    06-09    2906
    21-23    2184
    Name: hours, dtype: int64




```python
hours_FE10 = onehot_encode_single(hours_FE10, 'hours')
hours_FE10.drop(['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'],axis=1, inplace=True)
df_desc(hours_FE10)
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
      <th>dtype</th>
      <th>NAs</th>
      <th>Numerical</th>
      <th>Boolean</th>
      <th>Categorical</th>
      <th>Date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_temp</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>windspeed</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>total_bikes</th>
      <td>float64</td>
      <td>0</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_sunday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_monday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_tuesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_wednesday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_thursday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_friday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weekday_saturday</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>workingday_yes</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year_2011</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>year_2012</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_winter</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_spring</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_summer</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>season_fall</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_clear</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_mist</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_light_rain</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>weather_condition_heavy_rain</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_jan</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_feb</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_mar</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_apr</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_may</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_jun</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_jul</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_aug</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_sep</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_oct</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_nov</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>month_dec</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>predefined_peak</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>calculated_peak</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_00-05</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_06-09</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_10-15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_16-20</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_21-23</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
pipeline(hours_FE10, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  40 columns
    Features Train:  14491 items |  40 columns
    Features Test:    2888 items |  40 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.85
    
    Intercept: [1.81683856e+12]
    Coefficients: [[ 7.17291139e-02  9.64532837e-03 -1.63480377e-02 -7.61737824e+11
      -7.61737824e+11 -7.61737824e+11 -7.61737824e+11 -7.61737824e+11
      -7.61737824e+11 -7.61737824e+11  3.68436824e-03 -5.16544694e+11
      -5.16544694e+11 -6.08712290e+11 -6.08712290e+11 -6.08712290e+11
      -6.08712290e+11  1.49820423e+11  1.49820423e+11  1.49820423e+11
       1.49820423e+11  2.48979960e+10  2.48979960e+10  2.48979960e+10
       2.48979960e+10  2.48979960e+10  2.48979960e+10  2.48979960e+10
       2.48979960e+10  2.48979960e+10  2.48979960e+10  2.48979960e+10
       2.48979960e+10  2.07163272e-02  8.81482302e-02 -1.04562172e+11
      -1.04562172e+11 -1.04562172e+11 -1.04562172e+11 -1.04562172e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.83



```python
hours_FE_sel_lite = hours_FE10.copy()
```

This additional feature decrease slightly the results with the previously selected features, with a cross validation mean of 0.85 (0.89 for the previously selected features) and a metric of 0.87 (0.90 for the previously selected features). However, it decreases significantly the number of features from 59 to 40. The resulting dataset will be kept to be tested during the Features Selection phase.

### Outliers


```python
hours_outliers  = hours_FE_sel.copy()
```


```python
def remove_outliers(df, target, columns):
    x = df[columns]
    y = df[target]
    ols = sm.OLS(endog = y.astype(float), exog = x.astype(float))
    fit = ols.fit()
    test = fit.outlier_test()['bonf(p)']
    outliers = list(test[test<1e-3].index)
    df.drop(df.index[outliers], inplace = True)
    df.reset_index(drop=True, inplace = True)
    return df
```


```python
hours_outliers = remove_outliers(hours_outliers, 'total_bikes', ['actual_temp' , 'humidity', 'windspeed'])
```


```python
hours_outliers.shape
```




    (17379, 60)




```python
hours_FE_sel.equals(hours_outliers)
```




    True




```python
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('Dataset:', hours_FE_sel.shape[0],'rows |', hours_FE_sel.shape[1], 'columns'))
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('Dataset:', hours_outliers.shape[0],'rows |', hours_outliers.shape[1], 'columns'))
```

    Dataset:   17379 rows |  60 columns
    Dataset:   17379 rows |  60 columns


The numerical features have been tested for outliers, but no outlier has been found. The *lite* version of the dataset with binned hours will be proceeded as well for process consistency, even if no outlier should be found.


```python
hours_outliers_lite  = hours_FE_sel_lite.copy()
hours_outliers_lite = remove_outliers(hours_outliers_lite, 'total_bikes', ['actual_temp' , 'humidity', 'windspeed'])
hours_FE_sel_lite.equals(hours_outliers_lite)
```




    True



### Features Selection


```python
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('Dataset:', hours_outliers.shape[0],'rows |', hours_outliers.shape[1], 'columns'))
print()
pipeline(hours_outliers, 'total_bikes', lm, 10)
```

    Dataset:   17379 rows |  60 columns
    
    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  59 columns
    Features Train:  14491 items |  59 columns
    Features Test:    2888 items |  59 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [-5.35051488e+10]
    Coefficients: [[ 6.52898889e-02 -1.92426421e-02 -1.11274385e-02  9.53347837e+10
       9.53347837e+10  9.53347837e+10  9.53347837e+10  9.53347837e+10
       9.53347837e+10  9.53347837e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10 -1.12024906e+10 -1.12024906e+10
      -1.12024906e+10 -1.12024906e+10  1.71203613e-02 -8.77850532e+09
      -8.77850532e+09  1.48801267e+10  1.48801267e+10  1.48801267e+10
       1.48801267e+10 -5.02559899e+10 -5.02559899e+10 -5.02559899e+10
      -5.02559899e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  1.35272242e+10  1.35272242e+10  1.35272242e+10
       1.35272242e+10  5.19561768e-02  5.74054718e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.90


The dataset resulting from the Feature Engineering phase contains 59 features, with a model reaching the accuracy of 0.90. The Feature Selection phase aims to reduce the number of variables used by the model.

#### Recursive Feature Elimination

The Recursive Feature Elimination (RFE) method is used to select the most relevant features for the model. In order to find which number of features provides the best score, the dataset will be tested against each possibility.


```python
def pipeline_rfe_multi(df, target, algorithm, range_def, scores):
    features = list_features(df, target)
    X, X_train, X_test, y, y_train, y_test = train_test_split_0(df, target, features)
    print('Iterations:')
    for i in range_def:
        rfe = RFE(algorithm, i)
        rfe = rfe.fit(X, y.values.ravel())
        
        cols_rfe = list(X.loc[:, rfe.support_])
        X_rfe_sel = X_train[cols_rfe]
        X_rfe_test_sel = X_test[cols_rfe]

        algorithm.fit(X_rfe_sel, y_train)
        y_pred = algorithm.predict(X_rfe_test_sel)
        
        result_model = [i, '{:.2f}'.format(r2_score(y_test, y_pred)), cols_rfe]
        scores.loc[i] = result_model
        print(i, end='   ')
```


```python
scores = pd.DataFrame(columns=['features','score', 'cols'])
range_def = range(1, len(hours_outliers.columns))
pipeline_rfe_multi(hours_outliers, 'total_bikes', lm, range_def, scores)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  59 columns
    Features Train:  14491 items |  59 columns
    Features Test:    2888 items |  59 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Iterations:
    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   59   


```python
scores['features'] = scores['features'].astype('float')
scores['score'] = scores['score'].astype('float')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores['features'],
             y = scores['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_235_0.png)



```python
scores.nlargest(10, 'score')
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
      <th>features</th>
      <th>score</th>
      <th>cols</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>58.0</td>
      <td>0.90</td>
      <td>[actual_temp, humidity, weekday_sunday, weekda...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>59.0</td>
      <td>0.90</td>
      <td>[actual_temp, humidity, windspeed, weekday_sun...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>56.0</td>
      <td>0.89</td>
      <td>[actual_temp, weekday_sunday, weekday_monday, ...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>57.0</td>
      <td>0.89</td>
      <td>[actual_temp, weekday_sunday, weekday_monday, ...</td>
    </tr>
    <tr>
      <th>55</th>
      <td>55.0</td>
      <td>0.87</td>
      <td>[actual_temp, weekday_sunday, weekday_monday, ...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>54.0</td>
      <td>0.86</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>50.0</td>
      <td>0.74</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>51</th>
      <td>51.0</td>
      <td>0.74</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>52.0</td>
      <td>0.74</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>45</th>
      <td>45.0</td>
      <td>0.73</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
  </tbody>
</table>
</div>



The RFE doesn't reduce drastically the number of features without degrading the performance of the model.

#### Recursive Feature Elimination on *Lite* Dataset

The *lite* version using the binnbed hours will be tested.


```python
scores_lite = pd.DataFrame(columns=['features','score', 'cols'])
range_def = range(1, len(hours_outliers_lite.columns))
pipeline_rfe_multi(hours_outliers_lite, 'total_bikes', lm, range_def, scores_lite)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  40 columns
    Features Train:  14491 items |  40 columns
    Features Test:    2888 items |  40 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Iterations:
    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   


```python
scores_lite['features'] = scores_lite['features'].astype('float')
scores_lite['score'] = scores_lite['score'].astype('float')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores_lite['features'],
             y = scores_lite['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_242_0.png)



```python
scores_lite.nlargest(10, 'score')
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
      <th>features</th>
      <th>score</th>
      <th>cols</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>39</th>
      <td>39.0</td>
      <td>0.88</td>
      <td>[actual_temp, humidity, weekday_sunday, weekda...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37.0</td>
      <td>0.87</td>
      <td>[actual_temp, weekday_sunday, weekday_monday, ...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38.0</td>
      <td>0.87</td>
      <td>[actual_temp, humidity, weekday_sunday, weekda...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36.0</td>
      <td>0.84</td>
      <td>[actual_temp, weekday_sunday, weekday_monday, ...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35.0</td>
      <td>0.83</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40.0</td>
      <td>0.83</td>
      <td>[actual_temp, humidity, windspeed, weekday_sun...</td>
    </tr>
    <tr>
      <th>33</th>
      <td>33.0</td>
      <td>0.65</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34.0</td>
      <td>0.65</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>32</th>
      <td>32.0</td>
      <td>0.64</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31.0</td>
      <td>0.62</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
  </tbody>
</table>
</div>



The score can remain at 0.87 with only 38 features when the hours are binned.

#### Manual Selection

Some features seem to have a lower importance for the model. They will be manually removed from the dataset with binned hours to see how the score gets degrated:  
- The meteorological features would simplify the inputs for the model, if removed. Thus, `actual_temp`, `humidity`, `windspeed` and the `weather_condition` categorical variables will be removed.  
- The `workingday_yes` variable is also removed, as it most of the time duplicates information already conveyed by the weekdays and predefined peak variables - it would need to compromise with a loss of performance on holidays.  
- The seasons variables also most of the time are doublons with the months variables - not using them would only sacrifice on some performance for the few weeks  


```python
hours_manual = hours_outliers_lite.copy()
hours_manual.drop(['actual_temp', 'humidity', 'windspeed', 'weather_condition_clear',
                   'weather_condition_mist', 'weather_condition_light_rain', 'weather_condition_heavy_rain',
                   'workingday_yes',
                   'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
scores_manual = pd.DataFrame(columns=['features','score', 'cols'])
range_def = range(1, len(hours_manual.columns))
pipeline_rfe_multi(hours_manual, 'total_bikes', lm, range_def, scores_manual)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  28 columns
    Features Train:  14491 items |  28 columns
    Features Test:    2888 items |  28 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Iterations:
    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   


```python
scores_manual['features'] = scores_manual['features'].astype('float')
scores_manual['score'] = scores_manual['score'].astype('float')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores_manual['features'],
             y = scores_manual['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_250_0.png)



```python
scores_manual.nlargest(10, 'score')
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
      <th>features</th>
      <th>score</th>
      <th>cols</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28</th>
      <td>28.0</td>
      <td>0.85</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27.0</td>
      <td>0.82</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25.0</td>
      <td>0.61</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26.0</td>
      <td>0.61</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24.0</td>
      <td>0.60</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23.0</td>
      <td>0.57</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22.0</td>
      <td>0.51</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20.0</td>
      <td>0.04</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21.0</td>
      <td>0.04</td>
      <td>[weekday_sunday, weekday_monday, weekday_tuesd...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>-0.06</td>
      <td>[weekday_friday]</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(scores_manual.iloc[27]['cols']) - set(scores_manual.iloc[25]['cols'])
```




    {'calculated_peak', 'predefined_peak'}




```python
pipeline(hours_manual, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  28 columns
    Features Train:  14491 items |  28 columns
    Features Test:    2888 items |  28 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.83
    
    Intercept: [8.9212664e+11]
    Coefficients: [[ 4.98728704e+10  4.98728704e+10  4.98728704e+10  4.98728704e+10
       4.98728704e+10  4.98728704e+10  4.98728704e+10 -3.16109153e+11
      -3.16109153e+11 -4.13364934e+10 -4.13364934e+10 -4.13364934e+10
      -4.13364934e+10 -4.13364934e+10 -4.13364934e+10 -4.13364934e+10
      -4.13364934e+10 -4.13364934e+10 -4.13364934e+10 -4.13364934e+10
      -4.13364934e+10  3.53691218e-02  7.74599282e-02 -5.84553864e+11
      -5.84553864e+11 -5.84553864e+11 -5.84553864e+11 -5.84553864e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.85


Using a simpllified hours notation and keeping only information of time and peaks allow the model to reach a score of 0.85 with only 28 variables. It is also important to note that the two (2) additional features describing the utilization peaks allow the score to jump from 0.61 to 0.85.


```python
hours_manual_rain = hours_outliers_lite.copy()
hours_manual_rain.drop(['actual_temp', 'humidity', 'windspeed',
                        'workingday_yes',
                        'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
pipeline(hours_manual_rain, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  32 columns
    Features Train:  14491 items |  32 columns
    Features Test:    2888 items |  32 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.84
    
    Intercept: [7.48627682e+10]
    Coefficients: [[ 5.66553725e+10  5.66553725e+10  5.66553725e+10  5.66553725e+10
       5.66553725e+10  5.66553725e+10  5.66553725e+10 -1.19974544e+11
      -1.19974544e+11  3.25828055e+10  3.25828055e+10  3.25828055e+10
       3.25828055e+10 -5.86185689e+10 -5.86185689e+10 -5.86185689e+10
      -5.86185689e+10 -5.86185689e+10 -5.86185689e+10 -5.86185689e+10
      -5.86185689e+10 -5.86185689e+10 -5.86185689e+10 -5.86185689e+10
      -5.86185689e+10  3.68691505e-02  7.34777631e-02  1.44921664e+10
       1.44921664e+10  1.44921664e+10  1.44921664e+10  1.44921664e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.86


Adding back the rain information `weather_condition`, the model can integrate a simple weather input to slightly precise its predictions and reach a score of 0.86. The model now uses 32 features.


```python
hours_manual_full_weather = hours_outliers_lite.copy()
hours_manual_full_weather.drop(['workingday_yes',
                                'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
pipeline(hours_manual_full_weather, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  35 columns
    Features Train:  14491 items |  35 columns
    Features Test:    2888 items |  35 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.85
    
    Intercept: [-4.7568581e+10]
    Coefficients: [[ 6.95183492e-02 -2.27633705e-02 -8.16487881e-03 -2.37641347e+10
      -2.37641347e+10 -2.37641347e+10 -2.37641347e+10 -2.37641347e+10
      -2.37641347e+10 -2.37641347e+10  8.93798763e+09  8.93798763e+09
      -1.02829335e+10 -1.02829335e+10 -1.02829335e+10 -1.02829335e+10
       2.53374503e+10  2.53374503e+10  2.53374503e+10  2.53374503e+10
       2.53374503e+10  2.53374503e+10  2.53374503e+10  2.53374503e+10
       2.53374503e+10  2.53374503e+10  2.53374503e+10  2.53374503e+10
       3.78427191e-02  6.91086856e-02  4.73402113e+10  4.73402113e+10
       4.73402113e+10  4.73402113e+10  4.73402113e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


If a more precise meteorological information is taken in account, the model can reach a score of 0.87 with 35 features.


```python
hours_manual_hourly = hours_outliers.copy()
hours_manual_hourly.drop(['workingday_yes',
                          'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
pipeline(hours_manual_hourly, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  54 columns
    Features Train:  14491 items |  54 columns
    Features Test:    2888 items |  54 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.89
    
    Intercept: [-1.27616082e+11]
    Coefficients: [[ 6.23305418e-02 -1.77614159e-02 -1.20369352e-02  8.46865925e+10
       8.46865925e+10  8.46865925e+10  8.46865925e+10  8.46865925e+10
       8.46865925e+10  8.46865925e+10 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -3.60553097e+09 -3.60553097e+09
      -3.60553097e+09 -3.60553097e+09 -1.12915039e-02  1.12609863e-02
       8.26736053e+09  8.26736053e+09  8.26736053e+09  8.26736053e+09
       3.82676603e+10  3.82676603e+10  3.82676603e+10  3.82676603e+10
       3.82676603e+10  3.82676603e+10  3.82676603e+10  3.82676603e+10
       3.82676603e+10  3.82676603e+10  3.82676603e+10  3.82676603e+10
       5.03082275e-02  5.84564209e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.88


And if a precise hourly monitoring is required, the model can reach a score of 0.88 with 54 features.

## Final Metric

<font color='red'>__Confirm by changing the number of folds.__</font>

The Feature Selection phase suggested several acceptable models, which mainly differ on the number of features used and the resulting performance. Depending on the accuracy expected by the business, one of these models will be selected:  

1. Time and Utilization Peaks information only:  
Score: 0.85 | Features: 28    


2. Model with Simple Rain information:  
Score: 0.86 | Features: 32  


3. Model with Complete Weather information:  
Score: 0.87 | Features: 35  


4. Model with Hourly information:  
Score: 0.88 | Features: 54  


5. Complete Model:  
Score: 0.90 | Features: 59  

***

*Vratul Kapur | Irune Maury Arrue | Paul Jacques-Mignault | Sheena Miles | Ashley O’Mahony | Stavros Tsentemeidis | Karl Westphal  
O17 (Group G) | Master in Big Data and Business Analytics | Oct 2018 Intake | IE School of Human Sciences and Technology*

***
