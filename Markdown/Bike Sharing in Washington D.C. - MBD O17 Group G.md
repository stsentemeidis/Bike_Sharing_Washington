
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
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
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
# plt.axhline(hours_df.total_bikes.mean(), color='steelblue')
# plt.axhline(hours_df.total_bikes.mean()+0.315*hours_df.total_bikes.mean(), color='orange')
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


The season seems to have a consistent impact on the utilization by hour, with Winter discouraging a big part of the users.

#### Humidity by Actual Temperature


```python
plt.figure(figsize=(15,5))
sns.boxplot(x = hours_df.actual_temp,
            y = hours_df.humidity,
             color = 'steelblue')
plt.tight_layout()
```


![png](output_115_0.png)


The humidity seems to be changing regardless of the actual temperature, except for extreme temperature values where humidity seems to stabilize.

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
      <th>workingday_yes</th>
      <th>year_2011</th>
      <th>year_2012</th>
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
      <th>weekday_sunday</th>
      <th>weekday_monday</th>
      <th>weekday_tuesday</th>
      <th>weekday_wednesday</th>
      <th>weekday_thursday</th>
      <th>weekday_friday</th>
      <th>weekday_saturday</th>
      <th>season_winter</th>
      <th>season_spring</th>
      <th>season_summer</th>
      <th>season_fall</th>
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
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
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
      <td>1</td>
      <td>1</td>
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

    Intercept: [-7.83416043e+10]
    Coefficients: [[ 9.71226852e-02 -2.95181224e-02 -1.85827261e-02  9.45068384e-03
       2.95663726e+10  2.95663726e+10  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  3.38315350e+10  3.38315350e+10
       3.38315350e+10  3.38315350e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  1.44747179e+10  1.44747179e+10
       1.44747179e+10  1.44747179e+10  1.44747179e+10  1.44747179e+10
       1.44747179e+10 -3.31587495e+10 -3.31587495e+10 -3.31587495e+10
      -3.31587495e+10]]
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
from  matplotlib import pyplot
def pipeline(df, target, algorithm, n_splits = 10, plot=False):
    
    features = list_features(df, target)
    X, X_train, X_test, y, y_train, y_test = train_test_split_0(df, target, features)
    cross_val_ts(algorithm,X_train, y_train, n_splits)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    
    y_test_prep = pd.concat([y_test,pd.DataFrame(pd.DatetimeIndex(hours_df['date']).strftime('%Y-%m-%d'))], axis=1, sort=False, ignore_index=False)
    y_test_prep = y_test_prep.dropna()
    y_test_prep = y_test_prep.set_index(0)
    y_pred_prep = pd.DataFrame(y_pred)
    y_total_prep = pd.concat([y_test_prep.reset_index(drop=False), y_pred_prep.reset_index(drop=True)], axis=1)
    y_total_prep.columns = ['Date', 'Actual Total Bikes', 'Predicted Total Bikes']
    y_total_prep = y_total_prep.set_index('Date')

    print()
    print('Intercept:', lm.intercept_)
    print('Coefficients:', lm.coef_)
    print('Mean squared error (MSE): {:.2f}'.format(mean_squared_error(y_test, y_pred)))
    print('Variance score (R2): {:.2f}'.format(r2_score(y_test, y_pred)))
        
    if (plot == True):
        plt.figure(figsize=(15,5))
        g = sns.lineplot(data=y_total_prep, ci=None, lw=1, dashes=False)
        g.set_xticks(['2012-09-01', '2012-10-01', '2012-11-01', '2012-12-01', '2012-12-31'])
        g.legend(loc='lower left', ncol=1)
        plt.tight_layout()    
```


```python
pipeline(hours_clean, 'total_bikes', lm, 10, plot = True)
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
    
    Intercept: [-7.83416043e+10]
    Coefficients: [[ 9.71226852e-02 -2.95181224e-02 -1.85827261e-02  9.45068384e-03
       2.95663726e+10  2.95663726e+10  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  2.03371030e+09  2.03371030e+09
       2.03371030e+09  2.03371030e+09  3.38315350e+10  3.38315350e+10
       3.38315350e+10  3.38315350e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  3.15940180e+10  3.15940180e+10
       3.15940180e+10  3.15940180e+10  1.44747179e+10  1.44747179e+10
       1.44747179e+10  1.44747179e+10  1.44747179e+10  1.44747179e+10
       1.44747179e+10 -3.31587495e+10 -3.31587495e+10 -3.31587495e+10
      -3.31587495e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.76



![png](output_162_1.png)



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
    
    Intercept: [-8.27630428e+11]
    Coefficients: [[ 9.92133730e-02 -2.88428482e-02 -1.80268934e-02  1.07901538e-02
       5.96195705e+11  5.96195705e+11 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -2.67412253e+10 -2.67412253e+10
      -2.67412253e+10 -2.67412253e+10 -1.34252107e+11 -1.34252107e+11
      -1.34252107e+11 -1.34252107e+11  8.08195621e+11  8.08195621e+11
       8.08195621e+11  8.08195621e+11  8.08195621e+11  8.08195621e+11
       8.08195621e+11  8.08195621e+11  8.08195621e+11  8.08195621e+11
       8.08195621e+11  8.08195621e+11 -5.93776543e+11 -5.93776543e+11
      -5.93776543e+11 -5.93776543e+11 -5.93776543e+11 -5.93776543e+11
      -5.93776543e+11  2.80988151e+11  2.80988151e+11  2.80988151e+11
       2.80988151e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11
      -1.02979174e+11 -1.02979174e+11 -1.02979174e+11 -1.02979174e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.76


This additional feature gives the same result as the baseline, with a cross validation mean of 0.76 (0.75 for the baseline) and a metric of 0.76 (0.76 for the baseline). With the number of added variables and the lack of score improvement, we reject this feature.

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
    
    Intercept: [4.96112895e+11]
    Coefficients: [[ 1.09815303e-01 -3.00505364e-02 -6.97269315e-03  8.53324752e-04
      -2.19387418e+11 -2.19387418e+11 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10 -2.92721278e+10 -2.92721278e+10
      -2.92721278e+10 -2.92721278e+10  3.80970847e+10  3.80970847e+10
       3.80970847e+10  3.80970847e+10 -2.39193974e+10 -7.49608996e+10
      -1.09657587e+11 -1.52785513e+11 -8.35637910e+10 -1.01986464e+11
      -3.91089775e+10 -7.00312127e+10 -8.95897159e+10 -8.54036568e+10
      -9.02483368e+10 -1.27262650e+11  1.24140050e+11  1.24140050e+11
       1.24140050e+11  1.24140050e+11  1.24140050e+11  1.24140050e+11
       1.24140050e+11 -3.55542085e+11 -2.77336308e+11 -3.44361260e+11
      -3.35775084e+11 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
      -3.02290024e+10 -3.02290024e+10 -3.02290024e+10 -3.02290024e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  2.08124998e+10  2.08124998e+10  2.08124998e+10
       2.08124998e+10  5.55091872e+10  5.55091872e+10  5.55091872e+10
       5.55091872e+10  5.55091872e+10  5.55091872e+10  5.55091872e+10
       5.55091872e+10  5.55091872e+10  5.55091872e+10  5.55091872e+10
       5.55091872e+10  5.55091872e+10  5.55091872e+10  5.55091872e+10
       5.55091872e+10  5.55091872e+10  5.55091872e+10  5.55091872e+10
       5.55091872e+10 -2.26965900e+10 -2.26965900e+10 -2.26965900e+10
      -2.26965900e+10 -2.26965900e+10 -2.26965900e+10 -2.26965900e+10
      -2.26965900e+10 -2.26965900e+10 -2.26965900e+10 -2.26965900e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10  2.04313360e+10  2.04313360e+10
       2.04313360e+10  2.04313360e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -4.87903859e+10 -4.87903859e+10 -4.87903859e+10
      -4.87903859e+10 -3.03677130e+10 -3.03677130e+10 -3.03677130e+10
      -3.03677130e+10 -3.03677130e+10 -3.03677130e+10 -3.03677130e+10
      -3.03677130e+10 -3.03677130e+10 -3.03677130e+10 -3.03677130e+10
      -3.03677130e+10 -3.03677130e+10 -3.03677130e+10 -3.03677130e+10
      -3.03677130e+10 -3.03677130e+10 -3.03677130e+10 -3.03677130e+10
      -3.03677130e+10  3.66572390e+10  3.66572390e+10  3.66572390e+10
       3.66572390e+10  3.66572390e+10  3.66572390e+10  3.66572390e+10
       3.66572390e+10  3.66572390e+10  3.66572390e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10 -2.62202475e+10 -2.62202475e+10
      -2.62202475e+10 -2.62202475e+10  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  4.70198768e+09  4.70198768e+09  4.70198768e+09
       4.70198768e+09  2.42604909e+10  2.42604909e+10  2.42604909e+10
       2.42604909e+10  2.42604909e+10  2.42604909e+10  2.42604909e+10
       2.42604909e+10  2.42604909e+10  2.42604909e+10  2.42604909e+10
       2.42604909e+10  2.42604909e+10  2.42604909e+10  2.42604909e+10
       2.42604909e+10  2.42604909e+10  2.42604909e+10  2.42604909e+10
       2.42604909e+10  2.42604909e+10  2.42604909e+10  1.56743150e+10
       1.56743150e+10  1.56743150e+10  1.56743150e+10  1.56743150e+10
       1.56743150e+10  1.56743150e+10  1.56743150e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.14882559e+10  1.14882559e+10
       1.14882559e+10  1.14882559e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       1.63329359e+10  1.63329359e+10  1.63329359e+10  1.63329359e+10
       5.33472493e+10  5.33472493e+10  5.33472493e+10  5.33472493e+10
       5.33472493e+10  5.33472493e+10  5.33472493e+10  5.33472493e+10
       5.33472493e+10  5.33472493e+10  5.33472493e+10  5.33472493e+10
       5.33472493e+10  5.33472493e+10  5.33472493e+10  5.33472493e+10
       5.33472493e+10  5.33472493e+10  5.33472493e+10  5.33472493e+10
       7.31142504e+10  7.31142504e+10  7.31142504e+10  7.31142504e+10
       7.31142504e+10  7.31142504e+10  7.31142504e+10  7.31142504e+10
       7.31142504e+10  7.31142504e+10  7.31142504e+10]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.75


This additional feature gives a similar result than the baseline, with a cross validation mean of 0.78 (0.75 for the baseline) and a metric of 0.75 (0.76 for the baseline). With the number of added variables and the lack of score improvement, we reject this feature.

#### Calculated Peaks

The variable `calculated_peak` is added to flag the periods of typically high utilization.


```python
hours_FE3 = hours_FE_sel.copy()

#Set threshold to mean(Total_Bikes)+th based on Trial and Error appraoch
th = 0.315

def calculated_peaks(row, th):
    if (row['total_bikes'] > (hours_FE3.total_bikes.mean()+ th *hours_FE3.total_bikes.mean())):
        return 1
    else:
        return 0

hours_FE3['calculated_peak'] = hours_FE3.apply(lambda row: calculated_peaks(row, th), axis=1)
hours_FE3.calculated_peak.value_counts()
```




    0    11223
    1     6156
    Name: calculated_peak, dtype: int64




```python
pipeline(hours_FE3, 'total_bikes', lm, 10, plot = True)
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
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [5.1756512e+10]
    Coefficients: [[ 5.59973463e-02 -1.41487244e-02 -6.77050269e-03  8.18783830e-03
      -7.75197877e+10 -7.75197877e+10  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09 -2.82273060e+10 -2.82273060e+10
      -2.82273060e+10 -2.82273060e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10 -4.88030512e+09 -4.88030512e+09
      -4.88030512e+09 -4.88030512e+09 -4.88030512e+09 -4.88030512e+09
      -4.88030512e+09  2.55528049e+10  2.55528049e+10  2.55528049e+10
       2.55528049e+10  7.56726265e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87



![png](output_177_1.png)


This additional feature significantly improves the score of the model, with a cross validation mean of 0.86 (0.75 for the baseline) and a metric of 0.87 (0.76 for the baseline). Thus, we accept this feature.


```python
hours_FE_sel  = hours_FE3.copy()
```

#### Difference with Season Average Temperature

The variable `diff_season_avg_temp` is added to identify days with clement temperature compared to the seasonal average.


```python
hours_FE4 = hours_FE_sel.copy()

def diff_season_avg_temp_calc(df, row):
    if (row['season_spring'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_spring == 1].mean()
    
    elif (row['season_winter'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_winter == 1].mean()
    
    elif (row['season_fall'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_fall == 1].mean()
    
    elif (row['season_summer'] == 1):
        return row['actual_temp'] - df.actual_temp[df.season_summer == 1].mean()
    
hours_FE4['diff_season_avg_temp'] = hours_FE4.apply(lambda row: diff_season_avg_temp_calc(hours_FE4, row), axis=1)
```


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
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [-1.33806266e+10]
    Coefficients: [[-7.87681600e+07 -1.41540989e-02 -6.86824322e-03  8.21681321e-03
      -1.34840528e+09 -1.34840528e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.25756497e+09  1.25756497e+09
       1.25756497e+09  1.25756497e+09  1.68239194e+08  1.68239194e+08
       1.68239194e+08  1.68239194e+08 -1.41322185e+09 -1.41322185e+09
      -1.41322185e+09 -1.41322185e+09 -1.41322185e+09 -1.41322185e+09
      -1.41322185e+09 -1.41322185e+09 -1.41322185e+09 -1.41322185e+09
      -1.41322185e+09 -1.41322185e+09 -9.06296329e+09 -9.06296329e+09
      -9.06296329e+09 -9.06296329e+09 -9.06296329e+09 -9.06296329e+09
      -9.06296329e+09  2.38018494e+10  2.38215830e+10  2.38345835e+10
       2.38118153e+10  7.56715536e-02  7.87681601e+07]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.86 (0.86 for the previously selected features) and a metric of 0.87 (0.87 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Monthly Average Temperature

The variable `diff_month_avg_temp` is added to identify days with clement temperature compared to the monthly average.


```python
hours_FE5 = hours_FE_sel.copy()

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
    
hours_FE5['diff_month_avg_temp'] = hours_FE5.apply(lambda row: diff_month_avg_temp_calc(hours_FE5, row), axis=1)
```


```python
pipeline(hours_FE5, 'total_bikes', lm, 10)
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
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [9.7180024e+08]
    Coefficients: [[ 5.31599788e+06 -1.41446833e-02 -6.86142873e-03  8.21273541e-03
       2.36013643e+08  2.36013643e+08 -8.68099844e+07 -8.68099844e+07
      -8.68099844e+07 -8.68099844e+07 -8.68099844e+07 -8.68099844e+07
      -8.68099844e+07 -8.68099844e+07 -8.68099843e+07 -8.68099843e+07
      -8.68099844e+07 -8.68099844e+07 -8.68099843e+07 -8.68099843e+07
      -8.68099843e+07 -8.68099843e+07 -8.68099843e+07 -8.68099843e+07
      -8.68099843e+07 -8.68099843e+07 -8.68099843e+07 -8.68099844e+07
      -8.68099844e+07 -8.68099844e+07  8.93914649e+07  8.93914649e+07
       8.93914649e+07  8.93914649e+07  1.25400666e+08  1.25061309e+08
       1.24569340e+08  1.24140166e+08  1.23463312e+08  1.22978911e+08
       1.22591748e+08  1.22845426e+08  1.23345556e+08  1.24047079e+08
       1.24687069e+08  1.24931347e+08  6.02945483e+08  6.02945483e+08
       6.02945483e+08  6.02945483e+08  6.02945483e+08  6.02945483e+08
       6.02945483e+08 -1.93992214e+09 -1.93992214e+09 -1.93992214e+09
      -1.93992214e+09  7.56649338e-02 -5.31599782e+06]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.86 (0.86 for the previously selected features) and a metric of 0.87 (0.87 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Season Average Humidity

The variable `diff_season_avg_humi` is added to identify days with clement humidity compared to the seasonal average.


```python
hours_FE6 = hours_FE_sel.copy()

def diff_season_avg_humi_calc(df, row):
    if (row['season_spring'] == 1):
        return row['humidity'] - df.humidity[df.season_spring == 1].mean()
    
    elif (row['season_winter'] == 1):
        return row['humidity'] - df.humidity[df.season_winter == 1].mean()
    
    elif (row['season_fall'] == 1):
        return row['humidity'] - df.humidity[df.season_fall == 1].mean()
    
    elif (row['season_summer'] == 1):
        return row['humidity'] - df.humidity[df.season_summer == 1].mean()
    
hours_FE6['diff_season_avg_humi'] = hours_FE6.apply(lambda row: diff_season_avg_humi_calc(hours_FE6, row), axis=1)
```


```python
pipeline(hours_FE6, 'total_bikes', lm, 10)
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
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [-3.35645294e+10]
    Coefficients: [[ 5.59358809e-02  1.40125906e+11 -6.55312243e-03  7.98528815e-03
      -3.72715773e+10 -3.72715773e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10 -4.25851828e+10 -4.25851828e+10
      -4.25851828e+10 -4.25851828e+10  2.27667084e+10  2.27667084e+10
       2.27667084e+10  2.27667084e+10 -1.95870603e+09 -1.95870603e+09
      -1.95870603e+09 -1.95870603e+09 -1.95870603e+09 -1.95870603e+09
      -1.95870603e+09 -1.95870603e+09 -1.95870603e+09 -1.95870603e+09
      -1.95870603e+09 -1.95870603e+09 -3.80195466e+10 -3.80195466e+10
      -3.80195466e+10 -3.80195466e+10 -3.80195466e+10 -3.80195466e+10
      -3.80195466e+10  4.91708594e+10  4.27708076e+10  4.19096976e+10
       3.71514378e+10  7.57217407e-02 -1.40125906e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.86 (0.86 for the previously selected features) and a metric of 0.87 (0.87 for the previously selected features). With no clear score improvement, we reject this feature.

#### Difference with Monthly Average Humidity

The variable `diff_month_avg_humi` is added to identify days with clement humidity compared to the monthly average.


```python
hours_FE7 = hours_FE_sel.copy()

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
    
hours_FE7['diff_month_avg_humi'] = hours_FE7.apply(lambda row: diff_month_avg_humi_calc(hours_FE7, row), axis=1)
```


```python
pipeline(hours_FE7, 'total_bikes', lm, 10)
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
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [-5.91800322e+10]
    Coefficients: [[ 5.61264782e-02  1.60591351e+11 -6.88324035e-03  8.05493539e-03
       2.16924773e+10  2.16924773e+10 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -9.95534404e+09 -9.95534404e+09
      -9.95534404e+09 -9.95534404e+09 -5.62318429e+10 -5.62318429e+10
      -5.62318429e+10 -5.62318429e+10  1.79470167e+10  2.01502794e+10
       1.66432016e+10  1.67934239e+10  5.98716485e+08  1.87700740e+10
       1.52256962e+10  8.90630215e+09 -3.49562260e+09  5.13334310e+08
       1.08626076e+10  4.26700069e+09  8.99223193e+09  8.99223193e+09
       8.99223193e+09  8.99223193e+09  8.99223193e+09  8.99223193e+09
       8.99223193e+09 -1.65569563e+10 -1.65569563e+10 -1.65569563e+10
      -1.65569563e+10  7.57141113e-02 -1.60591351e+11]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.86 (0.86 for the previously selected features) and a metric of 0.87 (0.87 for the previously selected features). With no clear score improvement, we reject this feature.

#### Polynomial Features

The polynomial variables `actual_temp_poly_2`, `actual_temp_poly_3`, `humidity_poly_2`, `humidity_poly_3`, `windspeed_poly_2`, `windspeed_poly_3` are added to identify potential polynomial relationship between the target and the corresponding variables.


```python
hours_FE8 = hours_FE_sel.copy()
degree = 3
poly = PolynomialFeatures(degree)
list_var = ['actual_temp', 'humidity', 'windspeed']

def add_poly(df, degree, list_var):
    poly = PolynomialFeatures(degree)
    for var in list_var:
        poly_mat = poly.fit_transform(df[var].values.reshape(-1,1))
        for i in range(2, degree+1):
            df[var + '_poly_' + str(i)] = poly_mat[:,i]    

add_poly(hours_FE8, degree, list_var)
```


```python
pipeline(hours_FE8, 'total_bikes', lm, 10)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  64 columns
    Features Train:  14491 items |  64 columns
    Features Test:    2888 items |  64 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [-9.84474602e+09]
    Coefficients: [[-3.21837527e-02  7.95942577e-02  6.37548072e-03  7.47026760e-03
      -5.49132782e+09 -5.49132782e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09  5.99989553e+09  5.99989553e+09
       5.99989553e+09  5.99989553e+09 -2.32477219e+09 -2.32477219e+09
      -2.32477219e+09 -2.32477219e+09  1.78777052e+09  1.78777052e+09
       1.78777052e+09  1.78777052e+09  1.78777052e+09  1.78777052e+09
       1.78777052e+09  1.78777052e+09  1.78777052e+09  1.78777052e+09
       1.78777052e+09  1.78777052e+09  8.73952390e+07  8.73952390e+07
       8.73952390e+07  8.73952390e+07  8.73952390e+07  8.73952390e+07
       8.73952390e+07  9.78578475e+09  9.78578475e+09  9.78578475e+09
       9.78578475e+09  7.50973225e-02  2.85062671e-01 -2.34069109e-01
      -1.20605946e-01  4.09345627e-02 -1.73212439e-02 -5.37415594e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87


This additional feature gives a similar result than the previously selected features, with a cross validation mean of 0.86 (0.86 for the previously selected features) and a metric of 0.87 (0.87 for the previously selected features). With no clear score improvement, we reject this feature.

#### Hours Bins

The hour variables seem to be important for the model, but they are numerous. In order to reduce the number of variables, they will be binned into similar ranges.


```python
hours_FE9 = hours_FE_sel.copy()

def bin_hours(row):
    if (row['hour_0'] == 1) | (row['hour_1'] == 1) | (row['hour_2'] == 1) | (row['hour_3'] == 1) :
        return '00-03'
    
    if (row['hour_4'] == 1) | (row['hour_5'] == 1) | (row['hour_6'] == 1) | (row['hour_7'] == 1) :
        return '04-07'
    
    if (row['hour_8'] == 1) | (row['hour_9'] == 1) | (row['hour_10'] == 1) | (row['hour_11'] == 1):
        return '08-11'
    
    if (row['hour_12'] == 1) | (row['hour_13'] == 1) | (row['hour_14'] == 1) | (row['hour_15'] == 1): 
        return '12-15'
    
    if (row['hour_16'] == 1) | (row['hour_17'] == 1) |(row['hour_18'] == 1) | (row['hour_19'] == 1):
        return '16-19'
    
    if (row['hour_20'] == 1) | (row['hour_21'] == 1) | (row['hour_22'] == 1) | (row['hour_23'] == 1):
        return '20-23'

hours_FE9['hours'] = hours_FE9.apply(lambda row: bin_hours(row), axis=1)
hours_FE9.hours.value_counts()
```




    16-19    2916
    12-15    2915
    20-23    2912
    08-11    2908
    04-07    2866
    00-03    2862
    Name: hours, dtype: int64




```python
hours_FE9 = onehot_encode_single(hours_FE9, 'hours')
hours_FE9.drop(['hour_0', 'hour_1', 'hour_2', 'hour_3', 'hour_4', 'hour_5', 'hour_6', 'hour_7', 'hour_8', 'hour_9', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23'],axis=1, inplace=True)
df_desc(hours_FE9)
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
      <th>calculated_peak</th>
      <td>int64</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_00-03</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_04-07</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_08-11</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_12-15</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_16-19</th>
      <td>uint8</td>
      <td>0</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>hours_20-23</th>
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
pipeline(hours_FE9, 'total_bikes', lm, 10, plot = True)
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
    
    Cross Validation Variance score (R2): 0.81
    
    Intercept: [2.62416155e+09]
    Coefficients: [[ 4.85109637e-02 -1.26396133e-02 -2.69634454e-03  8.02223684e-03
       8.40622896e+09  8.40622896e+09 -7.93240203e+09 -7.93240203e+09
      -7.93240203e+09 -7.93240203e+09 -5.84581518e+09 -5.84581518e+09
      -5.84581518e+09 -5.84581518e+09 -5.84581518e+09 -5.84581518e+09
      -5.84581518e+09 -5.84581518e+09 -5.84581518e+09 -5.84581518e+09
      -5.84581518e+09 -5.84581518e+09 -2.70131280e+09 -2.70131280e+09
      -2.70131280e+09 -2.70131280e+09 -2.70131280e+09 -2.70131280e+09
      -2.70131280e+09  7.37892823e+09  7.37892823e+09  7.37892823e+09
       7.37892823e+09  8.87644794e-02 -1.92978872e+09 -1.92978872e+09
      -1.92978872e+09 -1.92978872e+09 -1.92978872e+09 -1.92978872e+09]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.83



![png](output_209_1.png)



```python
hours_FE_sel_lite = hours_FE9.copy()
```

This additional feature decreases slightly the results with the previously selected features, with a cross validation mean of 0.81 (0.86 for the previously selected features) and a metric of 0.83 (0.87 for the previously selected features). However, it decreases significantly the number of features from 59 to 40. The resulting dataset will be kept to be tested during the Features Selection phase.

### Features Selection


```python
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('Dataset:', hours_FE_sel.shape[0],'rows |', hours_FE_sel.shape[1], 'columns'))
print()
pipeline(hours_FE_sel, 'total_bikes', lm, 10, plot = True)
```

    Dataset:   17379 rows |  59 columns
    
    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  58 columns
    Features Train:  14491 items |  58 columns
    Features Test:    2888 items |  58 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [5.1756512e+10]
    Coefficients: [[ 5.59973463e-02 -1.41487244e-02 -6.77050269e-03  8.18783830e-03
      -7.75197877e+10 -7.75197877e+10  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09  3.03136927e+09  3.03136927e+09
       3.03136927e+09  3.03136927e+09 -2.82273060e+10 -2.82273060e+10
      -2.82273060e+10 -2.82273060e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10  3.02867126e+10  3.02867126e+10
       3.02867126e+10  3.02867126e+10 -4.88030512e+09 -4.88030512e+09
      -4.88030512e+09 -4.88030512e+09 -4.88030512e+09 -4.88030512e+09
      -4.88030512e+09  2.55528049e+10  2.55528049e+10  2.55528049e+10
       2.55528049e+10  7.56726265e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.87



![png](output_213_1.png)


The dataset resulting from the Feature Engineering phase contains 58 features, with a model reaching the accuracy of 0.87. The Feature Selection phase aims to reduce the number of variables used by the model.

#### Recursive Feature Elimination (Model A)

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
range_def = range(1, len(hours_FE_sel.columns))
pipeline_rfe_multi(hours_FE_sel, 'total_bikes', lm, range_def, scores)
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
    
    Iterations:
    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47   48   49   50   51   52   53   54   55   56   57   58   


```python
scores['features'] = scores['features'].astype('float')
scores['score'] = scores['score'].astype('float')
```


```python
scores.to_csv('Scores/scores_FE_sel.csv')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores['features'],
             y = scores['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_221_0.png)



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
      <th>55</th>
      <td>55.0</td>
      <td>0.87</td>
      <td>[actual_temp, year_2011, year_2012, hour_0, ho...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>56.0</td>
      <td>0.87</td>
      <td>[actual_temp, humidity, year_2011, year_2012, ...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>57.0</td>
      <td>0.87</td>
      <td>[actual_temp, humidity, workingday_yes, year_2...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>58.0</td>
      <td>0.87</td>
      <td>[actual_temp, humidity, windspeed, workingday_...</td>
    </tr>
    <tr>
      <th>54</th>
      <td>54.0</td>
      <td>0.86</td>
      <td>[year_2011, year_2012, hour_0, hour_1, hour_2,...</td>
    </tr>
    <tr>
      <th>52</th>
      <td>52.0</td>
      <td>0.73</td>
      <td>[year_2011, hour_0, hour_1, hour_2, hour_3, ho...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>53.0</td>
      <td>0.73</td>
      <td>[year_2011, year_2012, hour_0, hour_1, hour_2,...</td>
    </tr>
    <tr>
      <th>50</th>
      <td>50.0</td>
      <td>0.63</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
    <tr>
      <th>43</th>
      <td>43.0</td>
      <td>0.62</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
    <tr>
      <th>47</th>
      <td>47.0</td>
      <td>0.62</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(scores.iloc[57]['cols']) - set(scores.iloc[53]['cols'])
```




    {'actual_temp', 'humidity', 'windspeed', 'workingday_yes'}



The RFE offers to reduce the number of features to 54 without degrading significantly the performance of the model. The numerical weather variables and the working day flag are the first ones to be dropped by the RFE algorithm for a score of 0.86.


```python
hours_FE_sel_afterRFE = hours_FE_sel.copy()
```


```python
hours_FE_sel_afterRFE.drop(['actual_temp', 'humidity', 'windspeed', 'workingday_yes'],axis=1, inplace=True)
```


```python
print('{:<9} {:>6} {:>6} {:>3} {:>6}'.format('Dataset:', hours_FE_sel_afterRFE.shape[0],'rows |', hours_FE_sel_afterRFE.shape[1], 'columns'))
print()
pipeline(hours_FE_sel_afterRFE, 'total_bikes', lm, 10, plot = True)
```

    Dataset:   17379 rows |  55 columns
    
    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  54 columns
    Features Train:  14491 items |  54 columns
    Features Test:    2888 items |  54 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.85
    
    Intercept: [8.24511093e+10]
    Coefficients: [[-2.72751510e+08 -2.72751510e+08 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -1.48203104e+09 -1.48203104e+09
      -1.48203104e+09 -1.48203104e+09 -6.34783879e+08 -6.34783879e+08
      -6.34783879e+08 -6.34783879e+08 -6.94474204e+10 -6.94474204e+10
      -6.94474204e+10 -6.94474204e+10 -6.94474204e+10 -6.94474204e+10
      -6.94474204e+10 -6.94474204e+10 -6.94474204e+10 -6.94474204e+10
      -6.94474204e+10 -6.94474204e+10 -8.32490627e+09 -8.32490627e+09
      -8.32490627e+09 -8.32490627e+09 -8.32490627e+09 -8.32490627e+09
      -8.32490627e+09 -2.28921625e+09 -2.28921625e+09 -2.28921625e+09
      -2.28921625e+09  7.80782104e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.86



![png](output_227_1.png)


#### Recursive Feature Elimination on *Lite* Dataset

The *lite* version using the binned hours will be tested.


```python
scores_lite = pd.DataFrame(columns=['features','score', 'cols'])
range_def = range(1, len(hours_FE_sel_lite.columns))
pipeline_rfe_multi(hours_FE_sel_lite, 'total_bikes', lm, range_def, scores_lite)
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
scores_lite.to_csv('Scores/scores_FE_sel_lite.csv')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores_lite['features'],
             y = scores_lite['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_233_0.png)



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
      <td>0.83</td>
      <td>[actual_temp, humidity, workingday_yes, year_2...</td>
    </tr>
    <tr>
      <th>40</th>
      <td>40.0</td>
      <td>0.83</td>
      <td>[actual_temp, humidity, windspeed, workingday_...</td>
    </tr>
    <tr>
      <th>36</th>
      <td>36.0</td>
      <td>0.82</td>
      <td>[year_2011, year_2012, weather_condition_clear...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>37.0</td>
      <td>0.82</td>
      <td>[actual_temp, year_2011, year_2012, weather_co...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38.0</td>
      <td>0.82</td>
      <td>[actual_temp, humidity, year_2011, year_2012, ...</td>
    </tr>
    <tr>
      <th>34</th>
      <td>34.0</td>
      <td>0.63</td>
      <td>[year_2011, weather_condition_clear, weather_c...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>35.0</td>
      <td>0.63</td>
      <td>[year_2011, year_2012, weather_condition_clear...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31.0</td>
      <td>0.52</td>
      <td>[weather_condition_clear, weather_condition_mi...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18.0</td>
      <td>0.51</td>
      <td>[month_nov, weekday_sunday, weekday_monday, we...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19.0</td>
      <td>0.51</td>
      <td>[month_nov, month_dec, weekday_sunday, weekday...</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(scores_lite.iloc[39]['cols']) - set(scores_lite.iloc[35]['cols'])
```




    {'actual_temp', 'humidity', 'windspeed', 'workingday_yes'}



The score can be 0.82 with only 36 features. The numerical weather variables and the working day flag are the first ones to be dropped by the RFE algorithm.

#### Manual Selection (Model B)

Some features seem to have a lower importance for the model. They will be manually removed from the dataset to see how the score gets degrated:  
- The meteorological features would simplify the inputs for the model, if removed. Thus, `actual_temp`, `humidity`, `windspeed` and the `weather_condition` categorical variables will be removed.  
- The `workingday_yes` variable is also removed, as it most of the time duplicates information already conveyed by the weekdays and predefined peak variables - it would need to compromise with a loss of performance on holidays.  
- The seasons variables also most of the time are doublons with the months variables - not using them would only sacrifice on some performance for the few weeks  


```python
hours_manual = hours_FE_sel.copy()
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
    
    Features:        17379 items |  46 columns
    Features Train:  14491 items |  46 columns
    Features Test:    2888 items |  46 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Iterations:
    1   2   3   4   5   6   7   8   9   10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31   32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   


```python
scores_manual['features'] = scores_manual['features'].astype('float')
scores_manual['score'] = scores_manual['score'].astype('float')
```


```python
scores_manual.to_csv('Scores/scores_manual.csv')
```


```python
# Line Plot
plt.figure(figsize=(15,5))
sns.lineplot(x = scores_manual['features'],
             y = scores_manual['score'],
             color = 'steelblue')
plt.tight_layout()
```


![png](output_243_0.png)



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
      <th>45</th>
      <td>45.0</td>
      <td>0.85</td>
      <td>[year_2012, hour_0, hour_1, hour_2, hour_3, ho...</td>
    </tr>
    <tr>
      <th>46</th>
      <td>46.0</td>
      <td>0.85</td>
      <td>[year_2011, year_2012, hour_0, hour_1, hour_2,...</td>
    </tr>
    <tr>
      <th>44</th>
      <td>44.0</td>
      <td>0.83</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27.0</td>
      <td>0.59</td>
      <td>[hour_0, hour_1, hour_6, hour_7, hour_8, hour_...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28.0</td>
      <td>0.59</td>
      <td>[hour_0, hour_1, hour_2, hour_6, hour_7, hour_...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29.0</td>
      <td>0.59</td>
      <td>[hour_0, hour_1, hour_2, hour_5, hour_6, hour_...</td>
    </tr>
    <tr>
      <th>30</th>
      <td>30.0</td>
      <td>0.59</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_5, hour_...</td>
    </tr>
    <tr>
      <th>31</th>
      <td>31.0</td>
      <td>0.59</td>
      <td>[hour_0, hour_1, hour_2, hour_3, hour_4, hour_...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25.0</td>
      <td>0.58</td>
      <td>[hour_6, hour_7, hour_8, hour_9, hour_10, hour...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26.0</td>
      <td>0.58</td>
      <td>[hour_0, hour_6, hour_7, hour_8, hour_9, hour_...</td>
    </tr>
  </tbody>
</table>
</div>




```python
set(scores_manual.iloc[45]['cols']) - set(scores_manual.iloc[44]['cols'])
```




    {'year_2011'}




```python
pipeline(hours_manual, 'total_bikes', lm, 10, plot=True)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  46 columns
    Features Train:  14491 items |  46 columns
    Features Test:    2888 items |  46 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.84
    
    Intercept: [8.25783581e+11]
    Coefficients: [[-6.11237491e+07 -6.11237491e+07  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11  2.33864739e+11  2.33864739e+11
       2.33864739e+11  2.33864739e+11 -4.59212336e+10 -4.59212336e+10
      -4.59212336e+10 -4.59212336e+10 -4.59212336e+10 -4.59212336e+10
      -4.59212336e+10 -4.59212336e+10 -4.59212336e+10 -4.59212336e+10
      -4.59212336e+10 -4.59212336e+10 -1.01366596e+12 -1.01366596e+12
      -1.01366596e+12 -1.01366596e+12 -1.01366596e+12 -1.01366596e+12
      -1.01366596e+12  8.18704632e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.85



![png](output_246_1.png)


Using a simpllified hours notation and keeping only information of time and peaks allow the model to reach a score of 0.85 with only 46 variables. It is also important to note that the additional feature describing the utilization peaks allow the score to jump from 0.59 to 0.85.

#### Manual Selection with Rain information


```python
hours_manual_rain = hours_FE_sel.copy()
hours_manual_rain.drop(['actual_temp', 'humidity', 'windspeed',
                        'workingday_yes',
                        'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
pipeline(hours_manual_rain, 'total_bikes', lm, 10, plot=True)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  50 columns
    Features Train:  14491 items |  50 columns
    Features Test:    2888 items |  50 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.85
    
    Intercept: [1.73023464e+11]
    Coefficients: [[-5.99563624e+08 -5.99563624e+08  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11  2.70619086e+11  2.70619086e+11
       2.70619086e+11  2.70619086e+11 -8.34173259e+10 -8.34173259e+10
      -8.34173259e+10 -8.34173259e+10 -6.85177228e+10 -6.85177228e+10
      -6.85177228e+10 -6.85177228e+10 -6.85177228e+10 -6.85177228e+10
      -6.85177228e+10 -6.85177228e+10 -6.85177228e+10 -6.85177228e+10
      -6.85177228e+10 -6.85177228e+10 -2.91107938e+11 -2.91107938e+11
      -2.91107938e+11 -2.91107938e+11 -2.91107938e+11 -2.91107938e+11
      -2.91107938e+11  7.83386230e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.86



![png](output_250_1.png)


Adding back the rain information `weather_condition`, the model can integrate a simple weather input to slightly precise its predictions and reach a score of 0.86. The model now uses 50 features.

#### Manual Selection with Full Weather information


```python
hours_manual_full_weather = hours_FE_sel.copy()
hours_manual_full_weather.drop(['workingday_yes',
                                'season_fall', 'season_summer', 'season_spring', 'season_winter'],axis=1, inplace=True)
```


```python
pipeline(hours_manual_full_weather, 'total_bikes', lm, 10, plot=True)
```

    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  53 columns
    Features Train:  14491 items |  53 columns
    Features Test:    2888 items |  53 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Cross Validation Variance score (R2): 0.86
    
    Intercept: [3.9814623e+10]
    Coefficients: [[ 5.44413400e-02 -1.33412996e-02 -7.58973032e-03 -2.03399543e+10
      -2.03399543e+10  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09  2.57914770e+09  2.57914770e+09  2.57914770e+09
       2.57914770e+09 -6.92719137e+09 -6.92719137e+09 -6.92719137e+09
      -6.92719137e+09 -1.30465304e+10 -1.30465304e+10 -1.30465304e+10
      -1.30465304e+10 -1.30465304e+10 -1.30465304e+10 -1.30465304e+10
      -1.30465304e+10 -1.30465304e+10 -1.30465304e+10 -1.30465304e+10
      -1.30465304e+10 -2.08009466e+09 -2.08009466e+09 -2.08009466e+09
      -2.08009466e+09 -2.08009466e+09 -2.08009466e+09 -2.08009466e+09
       7.60688782e-02]]
    Mean squared error (MSE): 0.00
    Variance score (R2): 0.86



![png](output_254_1.png)


If a more precise meteorological information is taken in account, the model keeps the score of 0.86 with 53 features.

## Final Metric

The Feature Selection phase suggested several acceptable models, which mainly differ on the number of features used and the resulting performance. Depending on the accuracy expected by the business, one of these models will be selected:  

1. Model with Time and Peaks information - Model B:  
Score: 0.85 | Features: 46    


2. Model with Time, Peaks and Rain information:  
Score: 0.86 | Features: 50  


3. Model with Time, Peaks and Full Weather information and Binned Hours:  
Score: 0.86 | Features: 53  


4. Complete Model with Binned Hours:  
Score: 0.82 | Features: 36  


5. Complete Model - Model A:  
Score: 0.86 | Features: 54  


Based on the predictions and the way they fit the actual values, the models A and B are preferable.


```python
def pipeline_short(df, target, algorithm, n_splits = 10, plot=True):
    
    features = list_features(df, target)
    X, X_train, X_test, y, y_train, y_test = train_test_split_0(df, target, features)
#     cross_val_ts(algorithm,X_train, y_train, n_splits)
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    
    y_test_prep = pd.concat([y_test,pd.DataFrame(pd.DatetimeIndex(hours_df['date']).strftime('%Y-%m-%d'))], axis=1, sort=False, ignore_index=False)
    y_test_prep = y_test_prep.dropna()
    y_test_prep = y_test_prep.set_index(0)
    y_pred_prep = pd.DataFrame(y_pred)
    y_total_prep = pd.concat([y_test_prep.reset_index(drop=False), y_pred_prep.reset_index(drop=True)], axis=1)
    y_total_prep.columns = ['Date', 'Actual Total Bikes', 'Predicted Total Bikes']
    y_total_prep = y_total_prep.set_index('Date')

#     print()
#     print('Intercept:', lm.intercept_)
#     print('Coefficients:', lm.coef_)
#     print('Mean squared error (MSE): {:.2f}'.format(mean_squared_error(y_test, y_pred)))
    print('Variance score (R2): {:.2f}'.format(r2_score(y_test, y_pred)))
        
    if (plot == True):
        plt.figure(figsize=(15,5))
        g = sns.lineplot(data=y_total_prep, ci=None, lw=1, dashes=False)
        g.set_xticks(['2012-09-01', '2012-10-01', '2012-11-01', '2012-12-01', '2012-12-31'])
        g.legend(loc='lower left', ncol=1)
        plt.tight_layout()    
```


```python
print('#### Model A ####')
pipeline_short(hours_FE_sel_afterRFE, 'total_bikes', lm, 10, plot = True)
```

    #### Model A ####
    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  54 columns
    Features Train:  14491 items |  54 columns
    Features Test:    2888 items |  54 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Variance score (R2): 0.86



![png](output_259_1.png)



```python
print('#### Model B ####')
pipeline_short(hours_manual, 'total_bikes', lm, 10, plot=True)
```

    #### Model B ####
    Same indexes for X and y:                True
    Same indexes for X_train and y_train:    True
    Same indexes for X_test and y_test:      True
    
    Features:        17379 items |  46 columns
    Features Train:  14491 items |  46 columns
    Features Test:    2888 items |  46 columns
    Target:          17379 items |   1 columns
    Target Train:    14491 items |   1 columns
    Target Test:      2888 items |   1 columns
    
    Variance score (R2): 0.85



![png](output_260_1.png)


The Model A is better at predicting the peaks, which would ensure there will be no shortage and a better service level for the users.  

The Model B is much more simple, as it doesn't consider the weather parameters. It is still a good model and seems to be more stable with less variations.  

**A conservative approach would prefer the Model A to provide the best service to the bike sharing service users.**

## Further Work

A couple of ideas to improve the models and get them ready for production:

- The `Calculated Peaks` feature should be set up to use a moving average instead of a global mean. It would allow the model to adapt to the latest trend, especially if the utilization gets higher every year.
- More exploration of the `Day` and `Month-Day` features might be possible, if using the elapsed percentage of the month or different periods such as 4-weeks periods, to adapt to the local practices.
- Other algorithms could be applied on top of the linear regression to improve the model score and get more precise predictions. However, the benefits from a very precise model would need to be evaluated, as it might not be relevant compared to the response time of the operations and logistics of the bike sharing system.  

***

*Vratul Kapur | Irune Maury Arrue | Paul Jacques-Mignault | Sheena Miles | Ashley O’Mahony | Stavros Tsentemeidis | Karl Westphal  
O17 (Group G) | Master in Big Data and Business Analytics | Oct 2018 Intake | IE School of Human Sciences and Technology*

***
