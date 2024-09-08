# CryptoZinga Trading System. 
Cryptozinga showcases how machine learning can improve trade signals from a base technical analysis strategy. 
CryptoZinga uses feature extraction and extracts features that are only produced when a trading signal is found.

### Brief Overview
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
1.  The next step is to build lower stop loss, upper stop loss, lower take profit, and upper take profit for each candle!
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
