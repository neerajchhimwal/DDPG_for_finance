'''
use train and test sets 
    to get the best set of hyperparams 
        such that model trained on train set gives the best wins/loss ratio on test set

return this set of hyperparams 
    so actual model can be trained with these params on train+test set data 
        and traded on trade data (future data)
'''
#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime
import optuna
from pathlib import Path
from ddpg_torch import Agent
import gym
import numpy as np
import pandas as pd
from utils import sample_data_for_every_nth_day_of_the_month
from plot import get_comparison_df, backtest_stats, backtest_plot, get_daily_return, get_baseline
# from config import *

import config_tickers
from download_data import process_data
from stock_trading_env import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from trade_stocks import trade_on_test_df
# from finrl.agents.stablebaselines3.models import DRLAgent

import os
import sys
import ray
import kaleido
import itertools
import random
import torch

from config_tuning import *
from config import SEED, PERIOD, INDICATORS, DATE_OF_THE_MONTH_TO_TAKE_ACTIONS, BASELINE_TICKER_NAME_BACKTESTING
from hyp_utils import *

## Fixed
tpm_hist = {}  # record tp metric values for trials
tp_metric = 'avgwl'  # specified trade_param_metric: ratio avg value win/loss
## Settable by User
n_trials = N_TRIALS  # number of HP optimization runs
total_episodes = TOTAL_EPISODES # per HP optimization run
## Logging callback params
lc_threshold=1e-5
lc_patience=15
lc_trial_number=50


def sample_ddpg_params(trial:optuna.Trial):
    # Size of the replay buffer
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
#     tau = trial.suggest_categorical("tau", [0.01, 0.001, 0.1])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
#     learning_rate_critic = trial.suggest_loguniform("learning_rate_critic", 1e-5, 1e-1)
#     learning_rate_actor = 10**trial.suggest_int('logval', -5, 0)
#     learning_rate_critic = 10**trial.suggest_int('logval', -5, 0)
    
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "default": [400, 400],
        "big": [512, 512],
    }[net_arch]

    return {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "layer_1_size": net_arch[0],
            "layer_2_size": net_arch[1]
            }


def objective(trial:optuna.Trial, e_train_gym, env_kwargs, train_df, test_df):

    #Trial will suggest a set of hyperparamters from the specified range
    hyperparameters = sample_ddpg_params(trial)
    print(f'Hyperparameters from objective: {hyperparameters}')
    
    ACTOR_LR = hyperparameters["learning_rate"]
    CRITIC_LR = hyperparameters["learning_rate"]
    BATCH_SIZE = hyperparameters["batch_size"]
    TAU = 0.001
    LAYER_1_SIZE = hyperparameters["layer_1_size"]
    LAYER_2_SIZE = hyperparameters["layer_2_size"]
    buffer_size = hyperparameters["buffer_size"]
    
    model_ddpg = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=TUNING_TRIAL_MODELS_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=buffer_size,
              n_actions=stock_dimension)
    
    trained_ddpg = model_ddpg.train_model(
                        total_episodes=total_episodes, train_from_scratch=True, 
                        env=e_train_gym, env_kwargs=env_kwargs, save_ckp=False, ckp_save_freq=0,
                        use_wandb=False)
    
    if SAVE_MODELS:
        trained_ddpg.save_checkpoint(checkpoint_name='ddpg_{}.pt'.format(trial.number), last_episode=trained_ddpg.episode)
    
    #For the given hyperparamters, determine the account value in the trading period
    df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=test_df, 
                                                                             model=trained_ddpg, 
                                                                             train_df=train_df, 
                                                                             env_kwargs=env_kwargs,
                                                                            seed=SEED)

    # Calculate trade performance metric
    # Currently ratio of average win and loss market values
    tpm = calc_trade_perf_metric(df_actions, test_df, tp_metric, tpm_hist)
    return tpm

def get_tuned_hyperparams(train_df, test_df, env_kwargs):

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    # seed fixing for reproducability
    np.random.seed(SEED)
    e_train_gym.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    if SAVE_MODELS:
        os.makedirs(TUNING_TRIAL_MODELS_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name="ddpg_study", direction='maximize',
                                sampler=sampler, pruner=optuna.pruners.HyperbandPruner())

    logging_callback = LoggingCallback(threshold=lc_threshold,
                                        patience=lc_patience,
                                        trial_number=lc_trial_number)
    
    wrapper_obj_func = lambda trial: objective(trial, e_train_gym, env_kwargs, train_df, test_df)

    study.optimize(wrapper_obj_func, n_trials=n_trials, catch=(ValueError,), callbacks=[logging_callback])

    print('='*100)
    print('='*100)
    print('Hyperparameters after tuning', study.best_params)
    print('='*100)
    print('='*100)
    print(f'Study with trial number {study.best_trial.number} has the highest value: {study.best_trial.values[0]}')

    return study.best_params

if __name__ == "__main__":
    TRAIN_START_DATE = '2009-01-01'
    TRAIN_END_DATE = '2015-10-01'
    TEST_START_DATE = '2015-10-01'
    TEST_END_DATE = '2016-01-01'
    ticker_name_from_config_tickers = 'DOW_30_TICKER'
    TRAIN_CSV_NAME = f'./data/train_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.csv'
    TEST_CSV_NAME = f'./data/test_{ticker_name_from_config_tickers}_{TEST_START_DATE}_to_{TEST_END_DATE}.csv'

    print('reading csvs...')
    train = pd.read_csv(TRAIN_CSV_NAME, index_col='Unnamed: 0')
    test = pd.read_csv(TEST_CSV_NAME, index_col='Unnamed: 0')
    print(f'Train shape: {train.shape} \nTest shape: {test.shape}')

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    tuned_hyperparams = get_tuned_hyperparams(train_df=train, test_df=test, env_kwargs=env_kwargs)
    print('Tuned hyperparams: ', tuned_hyperparams)