import numpy as np
import pandas as pd

from gym.utils import seeding
import gym
from gym import spaces


import matplotlib
import matplotlib.pyplot as plt
import pickle

# Trading Environment Parameters:
# Maxium Shares per Trade : 100
# Initial Portfolio Balance: 1,000,000 USD
# Total Stocks to be traded : 30 from Dow 30
# The System Risk Indicatiors Threshold: Max value 150
# Transcation Cost and Fee Percentage : 0.1%


HMAX_NORMALIZE = 100

INITIAL_ACCOUNT_BALANCE=1000000

STOCK_DIM = 30

TRANSACTION_FEE_PERCENT = 0.001

TI_threshold = 150

REWARD_SCALING = 1e-4


class Train_Env(gym.Env):
   
    metadata = {'render.modes': ['human']}

    def __init__(self, df,day = 0):
        
         
        self.day = day
        self.df = df
       
        
        # Action Space Normalization, Shape: 30 (Stocks Number from Dow30)
        self.action_space = spaces.Box(low = -1, high = 1,shape = (STOCK_DIM,)) 
        
        # States Space Normalization, Shape : 
        # 1. Current Balance : Shape = 1
        # 2. Each Stock Price: Shape = 30
        # 3. Each Stock Share Number: Shape = 30 
        # 4. Each Stock Technical Indicators: Shape = MACD(30) + RSI(30) + CCI(30) + ADX(30) 
        # 
        # Total Shape  = 1 + 30 + 30 + 4*30 = 181
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (181,))
        
        
        # Initialization
        
        self.data = self.df.loc[self.day,:]
        self.terminal = False   
        
        
        
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist()
        
        
        
        self.reward = 0
        self.cost = 0
        
     
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
       
        self._seed()
       
    
    
    
    def _sell_stock(self, index, action):
        

        if self.state[index+STOCK_DIM+1] > 0:
            #update balance
            self.state[0] += self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * (1- TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            self.cost +=self.state[index+1]*min(abs(action),self.state[index+STOCK_DIM+1]) * TRANSACTION_FEE_PERCENT
            self.trades+=1
        else:
            pass

    
    def _buy_stock(self, index, action):
        
        available_amount = self.state[0] // self.state[index+1]
       
        self.state[0] -= self.state[index+1]*min(available_amount, action) *(1+ TRANSACTION_FEE_PERCENT)

        self.state[index+STOCK_DIM+1] += min(available_amount, action)

        self.cost+=self.state[index+1]*min(available_amount, action)* TRANSACTION_FEE_PERCENT
        
        self.trades+=1
       
    def step(self, actions):
       
        self.terminal = self.day >= len(self.df.index.unique())-1
       
        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/portfolio_value_train.png')
            plt.close()
            
            end_total_asset = self.state[0]+ sum(np.array(self.state[1:(STOCK_DIM+1)])*\
                                                 np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            
            step_total_value = pd.DataFrame(self.asset_memory)
            step_total_value.to_csv('results/portfolio_value_train.csv')
            
            # Calculate Sharpe Ratio
            
            step_total_value.columns = ['account_value']
            step_total_value['daily_return'] = step_total_value.pct_change(1)
            
            sharpe = (252**0.5)*step_total_value['daily_return'].mean()/step_total_value['daily_return'].std()
            
           
            
            return self.state, self.reward, self.terminal,{}

        else:
            

            actions = actions * HMAX_NORMALIZE
            
            begin_total_asset = self.state[0]+ sum(np.array(self.state[1:(STOCK_DIM+1)])*\
                                                   np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
           
            
            sell_index = np.argsort(actions)[:np.where(actions < 0)[0].shape[0]]
            buy_index = np.argsort(actions)[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                
                self._sell_stock(index, actions[index])

            for index in buy_index:
                
                self._buy_stock(index, actions[index])

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            
            #load next state
            
            self.state =  [self.state[0]] + \
                    self.data.close.values.tolist() + \
                    list(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]) + \
                    self.data.macd.values.tolist() + \
                    self.data.rsi.values.tolist() + \
                    self.data.cci.values.tolist() + \
                    self.data.adx.values.tolist()
            
            end_total_asset = self.state[0]+ sum(np.array(self.state[1:(STOCK_DIM+1)])*\
                                                 np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]))
            
            self.asset_memory.append(end_total_asset)
           
            self.reward = end_total_asset - begin_total_asset            
           
            self.rewards_memory.append(self.reward)
            
            self.reward = self.reward*REWARD_SCALING



        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False 
        self.rewards_memory = []
        #initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                      self.data.close.values.tolist() + \
                      [0]*STOCK_DIM + \
                      self.data.macd.values.tolist() + \
                      self.data.rsi.values.tolist() + \
                      self.data.cci.values.tolist() + \
                      self.data.adx.values.tolist() 
       
        return self.state
    
    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]