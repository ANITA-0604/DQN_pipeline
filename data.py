import os 
from pathlib import Path
import pandas as pd
import yfinance as yf
class DataLoader:
    def __init__(self, tickers, begin_date = None, end_date = None, interval = '1h'):
        
        self.tickers = tickers
        self.begin_date = begin_date
        self.end_date = end_date
        self.interval = interval
    
        
    def load_data(self):
        df = yf.download(self.tickers, start=self.begin_date, end=self.end_date, interval=self.interval)
        # df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.dropna(inplace=True)
        return df

        
        
    
