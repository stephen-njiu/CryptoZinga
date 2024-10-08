# -*- coding: utf-8 -*-
"""cross_over_ml.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YMYGscvJnsfx9fS8_zBco8wtK-M2Mzij

## Crossover Reinforcement using ML
* We set the shorter sma to 6
* We set the longer sma to 20
* We set the directional sma to determine general trend at 100
* We only trade at the direction of the market
* We dynamically set the stop loss and take profit using ATR. We do testing between setting the stop loss multiplier to 1 or 1.5.
* The take profit is hard coded to be 2X the stop loss to give a risk to reward ration of 1:2.
* 1. When there is a crossover which is above the sma_100, we go long
* 2. When there is a crossunder which is below the sma_100, we go short <br>
* <i> The two will serve as our baseline model for forecasting</i>
## What is the next Step
1.  The next step is to build lower stop loss, upper stop loss, lower take profit, and upper take profit for each candle!
2. For the crossovers that occur while meeting our defined crossover conditions, we check whether the market hits the stop loss or take profit first and return the corresponding value.
  * Returns 1 if we went long and the market hit the upper take profit
  * Returns 2 if we went short and the market hit the lower take profit
  * Returns 3 if both crossover conditions were well defined but stop losses were hit! (We return 3 instead of returning to 0)

## Next step
- We are setting these feature extraction after checking whether the take profits were hit to reduce processing power in the<i> long iteration process </i>
* We add other technical indicators to aid the `rfClassifier` in making decisions.
  * 1. The bollinger band width
  * 2. The ROC of the 3rd, and 9th candles. (2 columns)
  * 3. The adx with a period of 14 (lookback period)
  * 4. whether the candle body size is a momentum candle or not (Returns a boolean of int type).
  * 5. The body size of each candle
  <br>
  <i> Save this data into a copy `preserved dataset` to maintain consistency in time series data when new data needs to be appended to the bottom and feature extraction is needed from previous rows </i> <br>
  <br>
#### `Depending on how model reacts, we can choose to do negative shift of row values into columns to show how price and features transition. Can sometimes result into overfitting`

## STEPS TO FOLLOW
<i> FIRST, we drop all null rows </i>
1. Extracts the rows with crossover and their features and targets
2. divide the data into 75/25
  * Save the 25 data into another file to serve as external data.
  * Use the 75 for training and testing purpose (further splitting it into 75/25)
3. Use a rf classifier model to predict whether the model results to 1, 2, or 2.
  * 1 ==> the model predicted going long, and yes our upper tp was hit
  * 2 ==> the model predicted going short, and yes our lower tp was hit
  * 3 ==> The model predicted going either long or short and none of our take profits were hit.

4. Determine the accuracy of the model! Then load the external dataset and determine the accuracy.

## Final Step!
  - Trade implementation thru Brokers api (OANDA, DERIV) thru data streaming
  - Data is loaded from broker using provided broker api, passed thru data pipeline function `data_cleaning(df)` and appended to a preserved dataset.
  - We then check whether any crossover conditions were met. If yes, we take the transformed row and pass it to our rf model to validate the signal!
  <br> <br>
<i><b> The model only confirms a signal from our base prediction model (based on crossover and trend) whether it is valid or not. If valid, the trade is executed on the next open candle. Hardcoding a risk reward ration of 1:2 allows a 50% accuracy or more to be profitable </b></i>
"""

# install the necessary libraries
!pip install ta

# import the necessary libraries/modules
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## load the dataset!
df = pd.read_csv('EURUSD_M15.csv')
df.head()

len(df)

# Convert date to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df.set_index('Date', inplace=True)

# sma_6, sma_20 and sma_100 calculations
df['sma_6'] = ta.trend.SMAIndicator(df['Close'], window=6).sma_indicator()
df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
df['sma_100'] = ta.trend.SMAIndicator(df['Close'], window=100).sma_indicator()
df.tail(10)

# Next we calculate the bollinger band width using a period of 20 and std_dev_multiplier of 2.
# Besides the width, we also calculate the percentage of close value to the band
df['bb_width'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_wband()
df['bb_pband'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_pband()
df.tail(10)

# determine crossover and crossunder and whether they happen in the direction of the trend!
# Determine the trend
df['trend'] = np.where(df['Close'] > df['sma_100'], 'uptrend', 'downtrend')
# Check for crossover (SMA-6 crosses above SMA-20)
df['crossover'] = (df['sma_6'] > df['sma_20']) & (df['sma_6'].shift(1) <= df['sma_20'].shift(1))

# Check for crossunder (SMA-6 crosses below SMA-20)
df['crossunder'] = (df['sma_6'] < df['sma_20']) & (df['sma_6'].shift(1) >= df['sma_20'].shift(1))

# Only consider crossovers/crossunders in the direction of the trend
df['valid_cross'] = np.where(df['crossover'] & (df['trend'] == 'uptrend'),1,
                             np.where(df['crossunder'] & (df['trend'] == 'downtrend'),2,3))
# df.tail(10)

(df['valid_cross']==1).sum(), (df['valid_cross']==2).sum()

trades = (df['valid_cross']==1).sum() + (df['valid_cross']==2).sum()
trades

"""<i>The approach BELOW is quite solid for setting dynamic stop loss and take profit levels. It adapts to market volatility (through the use of ATR) and maintains a consistent risk-reward ratio.</i>**bold text**"""

# determine atr values, stop losses and take profits thresholds
stop_loss_multiplier = 1.0 # change between 1.0 and 1.5 and see which fits the need best!
risk_reward_ratio = 2.0
# Calculate the ATR (used to determine stop loss distance)
atr_period = 14
df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
# Calculate initial stop loss and take profit levels
df['LowerStopLoss'] = df['Low'] - (df['ATR'] * stop_loss_multiplier)
df['UpperStopLoss'] = df['High'] + (df['ATR'] * stop_loss_multiplier)

# Calculate the take profit based on risk-reward ratio (2x the distance from stop loss)
df['LowerTakeProfit'] = df['Close'] - ((df['UpperStopLoss'] - df['Close']) * risk_reward_ratio)
df['UpperTakeProfit'] = df['Close'] + ((df['Close'] - df['LowerStopLoss']) * risk_reward_ratio)
df.tail(10)

# Now determine whether when a trade is executed, does it turn into profits or not? Not that we are referring to the current take profit and stop loss value and we are determining if that will be hit by the next future candle. If crossover which are 1, take_profits are hit, return 1 else 0. For cross_under which are 2, return 2 if lower take profit is hit, else return 0.
# the new values are returned in a new colum trade_result