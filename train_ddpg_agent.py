from ddpg_torch import Agent
import gym
import numpy as np
import pandas as pd
from utils import plotLearning
import wandb
from config import *
import config_tickers
from download_data import process_data
from stock_trading_env import StockTradingEnv
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.preprocessor.preprocessors import data_split
from trade_stocks import trade_on_test_df
import os
from plot import get_comparison_df
import warnings
warnings.filterwarnings("ignore")
import random
import torch

# DATA
# df = download(TRAIN_START_DATE, TRADE_END_DATE, ticker_list=config_tickers.DOW_30_TICKER)
if not (os.path.exists(TRAIN_CSV_NAME) and os.path.exists(TEST_CSV_NAME) and os.path.exists(TRADE_CSV_NAME)):
    df = YahooDownloader(start_date = TRAIN_START_DATE,
                        end_date = TRADE_END_DATE,
                        ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
    print(f'Data Downloaded! shape:{df.shape}')

    df_processed = process_data(df, use_technical_indicator=True, technical_indicator_list=INDICATORS, 
                                use_vix=True, use_turbulence=True, user_defined_feature=False)

    print(f'Data Processed! shape:{df_processed.shape}')

    train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
    test = data_split(df_processed, TEST_START_DATE, TEST_END_DATE)
    trade = data_split(df_processed, TRADE_START_DATE, TRADE_END_DATE)

    print(f'Train shape: {train.shape} Test shape: {test.shape} Trade shape: {trade.shape}')
    print('Saving csvs...')
    df.to_csv(ORIGINAL_CSV_NAME)
    df_processed.to_csv(PROCESSED_CSV_NAME)
    train.to_csv(TRAIN_CSV_NAME)
    test.to_csv(TEST_CSV_NAME)
    trade.to_csv(TRADE_CSV_NAME)
else:
    print('reading csvs...')
    train = pd.read_csv(TRAIN_CSV_NAME, index_col='Unnamed: 0')
    trade = pd.read_csv(TRADE_CSV_NAME, index_col='Unnamed: 0')
    test = pd.read_csv(TEST_CSV_NAME, index_col='Unnamed: 0')
    print(f'Train shape: {train.shape} Test shape: {test.shape} Trade shape: {trade.shape}')

# env
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
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

env = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = env.get_sb_env()

ENV_NAME = 'stock_tr_scaled_actions'
w_config = dict(
  alpha = ACTOR_LR,
  beta = CRITIC_LR,
  batch_size = BATCH_SIZE,
  layer1_size = LAYER_1_SIZE,
  layer2_size = LAYER_2_SIZE,
  input_dims = state_space,
  n_actions = stock_dimension,
  tau = TAU,
  architecture = "DDPG",
  env = ENV_NAME,
  train_start_date = TRAIN_START_DATE,
  train_end_date = TRAIN_END_DATE,
  trade_start_date = TRADE_START_DATE,
  trade_end_date = TRADE_END_DATE,
  hmax = env_kwargs['hmax'],
  initial_amount = env_kwargs['initial_amount'],
  num_stock_shares = env_kwargs['num_stock_shares'],
  buy_cost_pct = env_kwargs['buy_cost_pct'],
  sell_cost_pct = env_kwargs['sell_cost_pct'],
  state_space = env_kwargs['state_space'],
  stock_dimension = env_kwargs['stock_dim'],
  tech_indicator_list = env_kwargs['tech_indicator_list'],
  action_space = env_kwargs['action_space'],
  reward_scaling = env_kwargs['reward_scaling'],
  train_csv = TRAIN_CSV_NAME,
  trade_csv = TRADE_CSV_NAME,
  seed = SEED,
  ticker_list_name = ticker_name_from_config_tickers
)

PROJECT_NAME = f"pytorch_tuned_sb_ddpg_{ENV_NAME.lower()}"
PROJECT_NAME = "ddpg_tuned_dji"

np.random.seed(SEED)
env.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# print('Asset: ', env.asset_memory)
# log

agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=CHECKPOINT_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=100000,
              n_actions=stock_dimension)

agent = agent.train_model(
                        total_episodes=TOTAL_EPISODES, train_from_scratch=TRAIN_FROM_SCRATCH, 
                        env=env, env_kwargs=env_kwargs, save_ckp=True, ckp_save_freq=SAVE_CKP_AFTER_EVERY_NUM_EPISODES,
                        use_wandb=USE_WANDB, wandb_config=w_config, wandb_project_name=PROJECT_NAME)

print('agent training complete')

df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=trade, model=agent, train_df=train, env_kwargs=env_kwargs)
# print('results table....')
# print(df_account_value.head())
results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING)
train_values = np.zeros(len(results_df))
train_values[list(results_df.metric).index('Cumulative returns')] = cumulative_rewards_per_step_this_episode[-1]
train_values[list(results_df.metric).index('Max drawdown')] = min(cumulative_rewards_per_step_this_episode)
results_df['train_data'] = train_values

# saving
results_dir = './results_after_tuning'
# results_dir = './results_lr_schedule_step_10_grad_clip_small_nw_400_400_2016_2022_may'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
account_value_csv_name = f'account_value_test_episode_{i}.csv'
actions_csv_name = f'daily_actions_test_episode_{i}.csv'
results_table_name = f'return_comparison_episode_{i}.csv'
df_account_value.to_csv(os.path.join(results_dir, account_value_csv_name))
df_actions.to_csv(os.path.join(results_dir, actions_csv_name))
results_df.to_csv(os.path.join(results_dir, results_table_name))
# logging
if USE_WANDB:
    res_table = wandb.Table(dataframe=results_df) 
    run.log({f'Results Episode {i}': res_table})





            