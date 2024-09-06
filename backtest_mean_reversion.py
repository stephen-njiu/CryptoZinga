import numpy as np
from backtesting import Backtest, Strategy

def numpy_sma(array, n):
    """Simple Moving Average using numpy"""
    return np.convolve(array, np.ones(n), 'valid') / n

def numpy_std(array, n):
    """Standard Deviation using numpy"""
    return np.std(np.lib.stride_tricks.sliding_window_view(array, n), axis=1)

def numpy_atr(high, low, close, n):
    """Average True Range using numpy"""
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    return np.convolve(tr, np.ones(n), 'valid') / n

def numpy_adx(high, low, close, n):
    """Average Directional Index using numpy"""
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    atr = np.convolve(tr, np.ones(n), 'valid') / n
    
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    pos_dm = np.where((up > down) & (up > 0), up, 0)[:-1]
    neg_dm = np.where((down > up) & (down > 0), down, 0)[:-1]
    
    pos_di = 100 * np.convolve(pos_dm, np.ones(n), 'valid') / atr / n
    neg_di = 100 * np.convolve(neg_dm, np.ones(n), 'valid') / atr / n
    
    dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
    adx = np.convolve(dx, np.ones(n), 'valid') / n
    
    # Pad the beginning of the array with NaNs to match the original array length
    return np.concatenate([np.full(len(high) - len(adx), np.nan), adx])

class MeanReversionStrategy(Strategy):
    adx_threshold = 25
    lookback_period = 20
    stddev_multiplier = 2.0
    stop_loss_multiplier = 1.0
    risk_reward_ratio = 2.0
    atr_period = 14

    def init(self):
        # Calculate Bollinger Bands
        close = np.array(self.data.Close)
        self.sma = self.I(numpy_sma, close, self.lookback_period)
        self.std = self.I(numpy_std, close, self.lookback_period)
        self.upper = self.I(lambda: self.sma + self.stddev_multiplier * self.std)
        self.lower = self.I(lambda: self.sma - self.stddev_multiplier * self.std)

        # Calculate ADX
        high, low, close = np.array(self.data.High), np.array(self.data.Low), np.array(self.data.Close)
        self.adx = self.I(numpy_adx, high, low, close, 14)

        # Calculate ATR
        self.atr = self.I(numpy_atr, high, low, close, self.atr_period)

    def next(self):
        price = self.data.Close[-1]
        
        # Check if the entire candle body is above/below Bollinger Bands
        above_upper = self.data.Open[-1] > self.upper[-1] and self.data.Close[-1] > self.upper[-1]
        below_lower = self.data.Open[-1] < self.lower[-1] and self.data.Close[-1] < self.lower[-1]

        # Generate signals
        short_signal = above_upper and self.adx[-1] < self.adx_threshold
        long_signal = below_lower and self.adx[-1] < self.adx_threshold

        # Calculate stop loss and take profit levels
        stop_loss_long = price - (self.atr[-1] * self.stop_loss_multiplier)
        take_profit_long = price + ((price - stop_loss_long) * self.risk_reward_ratio)
        
        stop_loss_short = price + (self.atr[-1] * self.stop_loss_multiplier)
        take_profit_short = price - ((stop_loss_short - price) * self.risk_reward_ratio)

        # Close existing positions if stop loss or take profit is hit
        for trade in self.trades:
            if trade.is_long:
                if self.data.Low[-1] <= trade.sl or self.data.High[-1] >= trade.tp:
                    trade.close()
            else:
                if self.data.High[-1] >= trade.sl or self.data.Low[-1] <= trade.tp:
                    trade.close()

        # Open new positions
        if not self.position:
            if long_signal:
                self.buy(sl=stop_loss_long, tp=take_profit_long)
            elif short_signal:
                self.sell(sl=stop_loss_short, tp=take_profit_short)

# Load and prepare data
import pandas as pd
df = pd.read_csv('EURUSD_M30.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M')
df.set_index('Date', inplace=True)

# Run backtest
bt = Backtest(df, MeanReversionStrategy, cash=10000, commission=.002)
stats = bt.run()

# Print results
print(stats)

# Optional: Plot the results
bt.plot()