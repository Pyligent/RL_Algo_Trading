
import pandas as pd
import numpy as np
import time
from stable_baselines.common.vec_env import DummyVecEnv

from model.model import *
import os

def trading_model() -> None:
    
   
    stock_data = pd.read_csv('dow30_dataset.csv', index_col=0)
    

    # Validation Date : 2015/10/01 
    
    val_trade_date = stock_data[(stock_data.datadate > 20151001)&(stock_data.datadate <= 20200101)].datadate.unique()
    print(val_trade_date)

    # Rolling Windows for 63 Trading days within Three Months Time Frame
    rebalance_window = 63
    validation_window = 63
    
    ## Ensemble Strategy
    trading_policy(df=stock_data, unique_trade_date= val_trade_date,
                          rebalance_window = rebalance_window,
                          validation_window=validation_window)


if __name__ == "__main__":
    trading_model()
