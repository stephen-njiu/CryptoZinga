import pandas as pd
import numpy as np
# import the necessary libraries/modules
import ta
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
print('successfully imported all the packages')

# constants
stop_loss_multiplier = 1 # change between 1.0 and 1.5
risk_reward_ratio = 2.0
# Calculate the ATR (used to determine stop loss distance)
atr_period = 14
adx_period = 14  # Common period for ADX is 14

df = pd.read_csv('BTCUSD_M15_TRAIN.csv')
len(df)



def data_cleaning(df):
  '''
  This function takes in a dataframe that must have the following columns: Open, High, Low, Close.
  '''
  if 'Vol' in df.columns:
    del df['Vol']
  if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
    df.set_index('Date', inplace=True)
  df['sma_6'] = ta.trend.SMAIndicator(df['Close'], window=6).sma_indicator()
  df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
  df['sma_100'] = ta.trend.SMAIndicator(df['Close'], window=100).sma_indicator()
  df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close'], window=adx_period).adx()
  df['bb_width'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_wband()
  df['bb_pband'] = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2).bollinger_pband()
  df['trend'] = np.where(df['Close'] > df['sma_100'], 1, 2)
# Check for crossover (SMA-6 crosses above SMA-20)
  df['crossover'] = ((df['sma_6'] > df['sma_20']) & (df['sma_6'].shift(1) <= df['sma_20'].shift(1))).astype(int)

# Check for crossunder (SMA-6 crosses below SMA-20)
  df['crossunder'] = ((df['sma_6'] < df['sma_20']) & (df['sma_6'].shift(1) >= df['sma_20'].shift(1))).astype(int)

# Only consider crossovers/crossunders in the direction of the trend
  df['valid_cross'] = np.where(df['crossover'],1,
                             np.where(df['crossunder'],2,3))
  df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=atr_period).average_true_range()
# Calculate initial stop loss and take profit levels
  df['LowerStopLoss'] = df['Low'] - (df['ATR'] * stop_loss_multiplier)
  df['UpperStopLoss'] = df['High'] + (df['ATR'] * stop_loss_multiplier)

  # Calculate the take profit based on risk-reward ratio (2x the distance from stop loss)
  df['LowerTakeProfit'] = df['Close'] - ((df['UpperStopLoss'] - df['Close']) * risk_reward_ratio)
  df['UpperTakeProfit'] = df['Close'] + ((df['Close'] - df['LowerStopLoss']) * risk_reward_ratio)
  df.dropna(inplace=True)

  return df

def evaluate_trades_precisely(df):
    df['trade_result'] = 3
    # df['trade_duration'] = 0
    i = 0

    while i < len(df):
        if df['valid_cross'].iloc[i] == 1:  # Long trade signal
            entry_price = df['Close'].iloc[i]
            stop_loss = df['LowerStopLoss'].iloc[i]
            take_profit = df['UpperTakeProfit'].iloc[i]

            for j in range(i+1, len(df)):
                if df['Low'].iloc[j] <= stop_loss:
                    df.loc[df.index[i], 'trade_result'] = 3  # Stop loss hit
                    # df.loc[df.index[i], 'trade_duration'] = j - i
                    # i = j  # Move outer loop to this point
                    break
                elif df['High'].iloc[j] >= take_profit:
                    df.loc[df.index[i], 'trade_result'] = 1  # Take profit hit
                    # df.loc[df.index[i], 'trade_duration'] = j - i
                    # i = j  # Move outer loop to this point
                    break
            else:
                # Trade didn't conclude within available data
                df.loc[df.index[i], 'trade_result'] = -1
                # df.loc[df.index[i], 'trade_duration'] = len(df) - i - 1

        elif df['valid_cross'].iloc[i] == 2:  # Short trade signal
            entry_price = df['Close'].iloc[i]
            stop_loss = df['UpperStopLoss'].iloc[i]
            take_profit = df['LowerTakeProfit'].iloc[i]

            for j in range(i+1, len(df)):
                if df['High'].iloc[j] >= stop_loss:
                    df.loc[df.index[i], 'trade_result'] = 3  # Stop loss hit
                    # df.loc[df.index[i], 'trade_duration'] = j - i
                    # i = j  # Move outer loop to this point to avoid repeating trades
                    break
                elif df['Low'].iloc[j] <= take_profit:
                    df.loc[df.index[i], 'trade_result'] = 2  # Take profit hit
                    # df.loc[df.index[i], 'trade_duration'] = j - i
                    # i = j  # Move outer loop to this point to avoid repeating trades
                    break
            else:
                # Trade didn't conclude within available data
                df.loc[df.index[i], 'trade_result'] = -1
                # df.loc[df.index[i], 'trade_duration'] = len(df) - i - 1

        i += 1  # Move to next candle if no trade was initiated

    return df

df = data_cleaning(df)
df = evaluate_trades_precisely(df)

df.head(20)

df['trade_result'].value_counts()

(df['valid_cross']==1).sum(), (df['valid_cross']==2).sum(), (df['valid_cross']==3).sum()

total_concluded_trades = ((df['valid_cross'] == 1) | (df['valid_cross'] == 2)).sum()
winning_trades = (df['trade_result'] == 1).sum() + (df['trade_result'] == 2).sum()
base_accuracy = winning_trades / total_concluded_trades
print(f"Total trade signal: {total_concluded_trades}")
# print(f"Winning trades: {winning_trades}")
print(f"Accuracy: {base_accuracy * 100}")

## Now using machine learning, how can we improve these results

ml = df[(df['valid_cross'] == 1) | (df['valid_cross'] == 2)]

# print len of ml
print(len(ml))
y = ml['trade_result']  # Target variable
x = ml.drop(columns=['trade_result'])  # Features: drop the target column

x

x.shape, y.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000, random_state=42)
# Train the model
model.fit(x_train, y_train)
# Make predictions
y_pred = model.predict(x_test)
# Evaluate the model
model_accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {100 * model_accuracy:.2f}')

## Testing on External data
test = pd.read_csv('BTCUSD_M15_TEST.csv')

test

df

test = data_cleaning(test)
test = evaluate_trades_precisely(test)
test = test[(test['valid_cross'] == 1) | (test['valid_cross'] == 2)]
test_data = test.drop(columns=['trade_result'])
test

test_data

new_y = test['trade_result']
new_y

new_pred_y = model.predict(test_data)
new_pred_y

ext_data_accuracy = accuracy_score(new_y, new_pred_y)
print(f'Accuracy: {ext_data_accuracy:.2f}')
print(len(new_pred_y))

