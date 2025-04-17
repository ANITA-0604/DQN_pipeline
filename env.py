import pandas as pd
import numpy as np
from typing import Tuple

class TradingEnv:
    def __init__(self, data: pd.DataFrame, window_size: int = 10, initial_balance: int = 10000) -> None:
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.reset()
        self.history = []


    def reset(self) -> np.ndarray:
        self.balance = self.initial_balance
        self.position = 0.0
        self.t = self.window_size
        self.done = False
        return self._get_observation_()


    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        
        price = float(self.data.iloc[self.t]["Close"].iloc[0])
        reward = 0
        
        if action == 0:  # Hold
            reward -= 0.01  # Small penalty for inaction
        elif action == 1 : # Buy
            if self.position == 0:
                self.position = self.balance / price
                self.balance = 0
        elif action == 2 : # Sell
            if self.position > 0:
                self.balance = self.position * price
                self.position = 0
                reward = self.balance - self.initial_balance
        
        self.t += 1
        self.done = self.t >= len(self.data)
        net_worth = self.balance + self.position * price
        self.history.append(net_worth)


        # print(f"Step: {self.t}, Action: {action}, Reward: {reward}")
        return self._get_observation_(), reward, self.done
                
    def get_history(self):
        return self.history
    
    def _get_observation_(self) -> np.ndarray:
        # extract a window of window_size rows
        # flatten it into a 1D array data and apply it as an input of a neural net
       return self.data.iloc[self.t - self.window_size:self.t].values.flatten() 




