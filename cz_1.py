# -*- coding: utf-8 -*-
"""CZ_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o-2gmtGinBbQ6YRxvY29tw6UCkod81Jr
"""

# !pip install ta

# import required libraries
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('EURUSD_M15.csv')
df.head()

df.info()

df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df.info()

df.set_index('Date', inplace=True)
df.head()

# now calculate the lower and upper bb band with a look back of 20, and a stddev of 2.0.
# calculate the body size of each candle
# calculate the maximum wick size of each candle

# Calculate the moving average (middle band)
lookback_period = 20
df['MiddleBand'] = df['Close'].rolling(window=lookback_period).mean()
# Calculate the standard deviation
df['StandardDeviation'] = df['Close'].rolling(window=lookback_period).std()
# Calculate the upper and lower Bollinger Bands
stddev_multiplier = 2.0
df['UpperBand'] = df['MiddleBand'] + (stddev_multiplier * df['StandardDeviation'])
df['LowerBand'] = df['MiddleBand'] - (stddev_multiplier * df['StandardDeviation'])

df['BodySize'] = abs(df['Close'] - df['Open'])

# Calculate the upper wick size
df['UpperWickSize'] = df['High'] - df[['Open', 'Close']].max(axis=1)

# Calculate the lower wick size
df['LowerWickSize'] = df[['Open', 'Close']].min(axis=1) - df['Low']

# Calculate the maximum wick size
df['MaxWickSize'] = df[['UpperWickSize', 'LowerWickSize']].max(axis=1)

df.head(30)

# check whether the body_size is greater than max wick size
df['momentum'] = (df['BodySize'] >= df['MaxWickSize']).astype(int)
df.head(30)

df.columns



# for each candle, check whether the whole body is greater than the upper_bollinger_band and if true return 1, whether the whole body is less than the lower bollinger
# band and return 2 and else return 0
# Check conditions and assign values

df['BodyVsBollinger'] = 0

# Check if both Open and Close are above the Upper Band
df.loc[(df['Open'] >= df['UpperBand']) & (df['Close'] >= df['UpperBand']), 'BodyVsBollinger'] = 1

# Check if both Open and Close are below the Lower Band
df.loc[(df['Open'] <= df['LowerBand']) & (df['Close'] <= df['LowerBand']), 'BodyVsBollinger'] = 2

# Display the updated DataFrame
df.head(30)

(df['BodyVsBollinger'] == 1).sum(), (df['BodyVsBollinger'] == 2).sum()

# first we compute the profitability of the system based on manipulation without considering the wick of the body
df['pureSignal'] = np.where((df['momentum']==1) & (df['BodyVsBollinger']==1), 1,
                            np.where((df['momentum']==1) & (df['BodyVsBollinger']==2), 2, 0))
df.head(30)

(df['pureSignal'] == 1).sum(), (df['pureSignal'] == 2).sum()

"""## ADD Trade Parameters
* Calculate the upper stop loss and the lower stop loss. Also, calculate the upper take profit and the lower take profit.
* To calculate the stop loss value, we use the atr based method. We will define the atr multiplier to get the stop loss of both upper and below!

"""

# Calculate ATR
atr_period = 14  # ATR period
df['ATR'] = ta.volatility.AverageTrueRange(
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    window=atr_period
).average_true_range()
# Define a multiplier for the stop loss
stop_loss_multiplier = 1.5
df['LowerStopLoss'] = df['Low'] - (df['ATR'] * stop_loss_multiplier)
df['UpperStopLoss'] = df['High'] + (df['ATR'] * stop_loss_multiplier)

"""<i> Now that we have the upper and lower stop loss, our next task is to dynamically add the upper and lower stop loss ==> an even more challenging task.
* For lower take profit, we  take upperstoploss - df.close. We then multiply this distance by risk-reward-ratio. We then take the close value of the candle and subtract this distance.

* For upper take profit, we take the close - lower stop loss and multiply this value by risk-reward_multiplier. we then take the close and add this value.
"""

risk_reward_ratio = 2.09
df['LowerTakeProfit'] = df['Close'] - ((df['UpperStopLoss'] - df['Close']) * risk_reward_ratio)
df['UpperTakeProfit'] = df['Close'] + ((df['Close'] - df['LowerStopLoss']) * risk_reward_ratio)
df.head(30)

ss_trades, ll_trades = (df['BodyVsBollinger']==1).sum(), (df['BodyVsBollinger']==2).sum()
ss_trades, ll_trades # when we have not added the momentum factor into our trades...

short_trades, long_trades = (df['BodyVsBollinger'] == 1).sum(), (df['BodyVsBollinger'] == 2).sum()

short_trades, long_trades

"""* <i> now that we have everything that we need to take our signals, it is time to evaluate whether when 1, the trades hits take profit or the trade hits stop loss. Same when the signal is 2. Does the trade hits the upper take profit or does it hit the lower stop loss </i>

"""

# drop all null values
df.dropna(inplace=True)
df[:30]

def calculate_final_result(df):
    df['final_result'] = 0  # Initialize with 0

    for i in range(len(df)):
        current_index = df.index[i]
        if df.loc[current_index, 'BodyVsBollinger'] == 1:
            for j in range(i+1, len(df)):
                future_index = df.index[j]
                if df.loc[future_index, 'Low'] <= df.loc[current_index, 'LowerTakeProfit']:
                    df.loc[current_index, 'final_result'] = 1
                    break
                elif df.loc[future_index, 'High'] >= df.loc[current_index, 'UpperStopLoss']:
                    df.loc[current_index, 'final_result'] = 0
                    break
        elif df.loc[current_index, 'BodyVsBollinger'] == 2:
            for j in range(i+1, len(df)):
                future_index = df.index[j]
                if df.loc[future_index, 'High'] >= df.loc[current_index, 'UpperTakeProfit']:
                    df.loc[current_index, 'final_result'] = 2
                    break
                elif df.loc[future_index, 'Low'] <= df.loc[current_index, 'LowerStopLoss']:
                    df.loc[current_index, 'final_result'] = 0
                    break

    return df

# Assuming your DataFrame is called 'df'
df = calculate_final_result(df)

df

c_shorts, c_longs = (df['final_result']==1).sum(), (df['final_result']==2).sum()
c_shorts, c_longs

c_shorts/short_trades, c_longs/long_trades

