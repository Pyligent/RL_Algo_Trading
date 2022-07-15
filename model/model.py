
import pandas as pd
import numpy as np
import time,datetime
import datetime
import os


# RL models from stable-baselines
# 2021
import gym
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv


from env.Train_Env import Train_Env
from env.Val_Env import Val_Env
from env.Trading_Env import Trading_Env


def train_A2C(env_train, model_name, timesteps=25000):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()


    now = datetime.datetime.now()
    TRAINED_MODEL_DIR = f"trained_models/{now}"

    model.save(f"{TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model





def RL_Trading_Prediction(df,model,name,last_state,iter_num, unique_trade_date,rebalance_window,turbulence_threshold,initial):
  

    
    start=unique_trade_date[iter_num - rebalance_window]
    end=unique_trade_date[iter_num]
    trade_data = df[(df.datadate >= start) & (df.datadate < end)]
    trade_data = trade_data.sort_values(['datadate','tic'],ignore_index=True)
    trade_data.index = trade_data.datadate.factorize()[0]
    
    env_trade = DummyVecEnv([lambda: Trading_Env(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=initial,
                                                   previous_state=last_state,
                                                   model_name=name,
                                                   iteration=iter_num)])
    obs_trade = env_trade.reset()
    

    for i in range(len(trade_data.index.unique())):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_trade.step(action)
        if i == (len(trade_data.index.unique()) - 2):
            last_state = env_trade.render()

    df_last_state = pd.DataFrame({'last_state': last_state})
    df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
    
    return last_state


def RL_Trading_Val(model, test_data, test_env, test_obs) -> None:
  
    for i in range(len(test_data.index.unique())):
        action, _states = model.predict(test_obs)
        test_obs, rewards, dones, info = test_env.step(action)


def get_validation_sharpe(iteration):
    
    
    df_total_value = pd.read_csv('results/portfolio_value_validation_{}.csv'.format(iteration), index_col=0)
    df_total_value.columns = ['account_value_train']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
    return sharpe


def trading_policy(df, unique_trade_date, rebalance_window, validation_window) -> None:
   

    last_state = []

   
    a2c_sharpe_list = []

    model_use = []
    
    # Determine the turbulence_threshold from 2009 - 2020

   
    insample_turbulence = df[(df.datadate<20151000) & (df.datadate>=20090000)]
    insample_turbulence = insample_turbulence.drop_duplicates(subset=['datadate'])
    
    insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

    start = time.time()
    for i in range(rebalance_window + validation_window, len(unique_trade_date), rebalance_window):
        print("============================================")
        ## initial state is empty
        if i - rebalance_window - validation_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        # Tuning trubulence index based on historical data
        # Turbulence lookback window is one quarter
        
        end_date_index = df.index[df["datadate"] == unique_trade_date[i - rebalance_window - validation_window]].to_list()[-1]
        start_date_index = end_date_index - validation_window*30 + 1

        historical_turbulence = df.iloc[start_date_index:(end_date_index + 1), :]
       

        historical_turbulence = historical_turbulence.drop_duplicates(subset=['datadate'])

        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        if historical_turbulence_mean > insample_turbulence_threshold:
            
            turbulence_threshold = insample_turbulence_threshold
        else:
            
            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            
        print("turbulence_threshold: ", turbulence_threshold)

        
        # Training Setting 
        

        train_start = 20090000
        train_end = unique_trade_date[i - rebalance_window - validation_window]
        train = df[(df.datadate >= train_start) & (df.datadate < train_end)]
        train = train.sort_values(['datadate','tic'],ignore_index=True)
        train.index = train.datadate.factorize()[0]
       
        env_train = DummyVecEnv([lambda: Train_Env(train)])

        # Validation Setting

        val_start = unique_trade_date[i - rebalance_window - validation_window]
        val_end = unique_trade_date[i - rebalance_window]
        validation = df[(df.datadate >= val_start) & (df.datadate < val_end)]
        validation = validation.sort_values(['datadate','tic'],ignore_index=True)
        validation.index = validation.datadate.factorize()[0]

       
        env_val = DummyVecEnv([lambda: Val_Env(validation,
                                                          turbulence_threshold=turbulence_threshold,
                                                          iteration=i)])
        obs_val = env_val.reset()
        
       
    
        # Training 
        print("======Model training from: ", 20090000, "to ",
              unique_trade_date[i - rebalance_window - validation_window])
       
        print("======A2C Training========")
        model_a2c = train_A2C(env_train, model_name="A2C_30k_dow_{}".format(i), timesteps=30000)
        print("======A2C Validation from: ", unique_trade_date[i - rebalance_window - validation_window], "to ",
              unique_trade_date[i - rebalance_window])
        RL_Trading_Val(model=model_a2c, test_data=validation, test_env=env_val, test_obs=obs_val)
        sharpe_a2c = get_validation_sharpe(i)
        print("A2C Sharpe Ratio: ", sharpe_a2c)

        
       
        a2c_sharpe_list.append(sharpe_a2c)
        model= model_a2c
                
            
        # Trading

        print("======Trading from: ", unique_trade_date[i - rebalance_window], "to ", unique_trade_date[i])
        #print("Used Model: ", model_ensemble)
        last_state = RL_Trading_Prediction(df=df, model=model, name="A2C",
                                             last_state=last_state, iter_num=i,
                                             unique_trade_date=unique_trade_date,
                                             rebalance_window=rebalance_window,
                                             turbulence_threshold=turbulence_threshold,
                                             initial=initial)
        
       

    end = time.time()
    print("A2C Trading Strategy Time: ", (end - start) / 60, " minutes")
