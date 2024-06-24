import gymnasium as gym
import gym_trading_env
import pandas as pd
import numpy as np
from ta import add_all_ta_features

np.random.seed(69)

class StockTradingEnvironment:
    """This class wraps the gym-trading-env environment."""

    def __init__(self, df, train=True, number_of_days_to_consider=20, positions=[-1, 0, 1], windows=None, trading_fees=0, borrow_interest_rate=0, portfolio_initial_value=1000, initial_position='random', max_episode_duration='max', verbose=1):
        self.df = df
        self.train = train
        self.number_of_days_to_consider = number_of_days_to_consider
        self.positions = positions
        self.windows = windows
        self.trading_fees = trading_fees
        self.borrow_interest_rate = borrow_interest_rate
        self.portfolio_initial_value = portfolio_initial_value
        self.initial_position = initial_position
        self.max_episode_duration = max_episode_duration
        self.verbose = verbose

        # Adding technical indicators
        self.df = self._add_technical_indicators(self.df)
        print("Columns after adding technical indicators:", self.df.columns)  # Debugging line

        # Initialize the gym-trading-env environment
        self.env = gym.make('TradingEnv', df=self.df, positions=self.positions, windows=self.windows, trading_fees=self.trading_fees, borrow_interest_rate=self.borrow_interest_rate, portfolio_initial_value=self.portfolio_initial_value, initial_position=self.initial_position, max_episode_duration=self.max_episode_duration, verbose=self.verbose)
        self.action_space = self.env.action_space

        # Add custom metrics
        self.env.unwrapped.add_metric('Position Changes', lambda history: np.sum(np.diff(history['position']) != 0))
        self.env.unwrapped.add_metric('Episode Length', lambda history: len(history['position']))

    def _add_technical_indicators(self, df):
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
        df.fillna(0, inplace=True)
        return df

    def reset(self):
        observation, info = self.env.reset()
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        self.env.render()

    def close(self):
        self.env.close()

def make_env(file_path, **kwargs):
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    return StockTradingEnvironment(df, **kwargs).env