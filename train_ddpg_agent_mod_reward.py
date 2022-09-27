from ddpg_torch import Agent
import numpy as np
import pandas as pd
from utils import get_baseline_daily_returns
import wandb
from config import *
import config_tickers
from download_data import process_data
from stock_trading_env_mod_reward import StockTradingEnv
# from env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from trade_stocks import trade_on_test_df
import os
from plot import get_baseline_ind, get_comparison_df, backtest_plot, get_baseline, get_daily_return
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import matplotlib.pyplot as plt
import empyrical as ep
from utils import sample_data_for_every_nth_day_of_the_month
import yfinance as yf
from copy import deepcopy

# DATA
# df = download(TRAIN_START_DATE, TRADE_END_DATE, ticker_list=config_tickers.DOW_30_TICKER)
# if not (os.path.exists(TRAIN_CSV_NAME) and os.path.exists(TEST_CSV_NAME) and os.path.exists(TRADE_CSV_NAME)):
#     df = YahooDownloader(start_date = TRAIN_START_DATE,
#                         end_date = TRADE_END_DATE,
#                         ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
#     print(f'Data Downloaded! shape:{df.shape}')

#     df_processed = process_data(df, use_technical_indicator=True, technical_indicator_list=INDICATORS, 
#                                 use_vix=True, use_turbulence=True, user_defined_feature=False)

#     print(f'Data Processed! shape:{df_processed.shape}')

#     train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
#     test = data_split(df_processed, TEST_START_DATE, TEST_END_DATE)
#     trade = data_split(df_processed, TRADE_START_DATE, TRADE_END_DATE)

#     print(f'Train shape: {train.shape} Test shape: {test.shape} Trade shape: {trade.shape}')
#     print('Saving csvs...')
#     df.to_csv(ORIGINAL_CSV_NAME)
#     df_processed.to_csv(PROCESSED_CSV_NAME)
#     train.to_csv(TRAIN_CSV_NAME)
#     test.to_csv(TEST_CSV_NAME)
#     trade.to_csv(TRADE_CSV_NAME)
# else:
#     print('reading csvs...')
#     train = pd.read_csv(TRAIN_CSV_NAME, index_col='Unnamed: 0')
#     trade = pd.read_csv(TRADE_CSV_NAME, index_col='Unnamed: 0')
#     test = pd.read_csv(TEST_CSV_NAME, index_col='Unnamed: 0')
#     print(f'Train shape: {train.shape} Test shape: {test.shape} Trade shape: {trade.shape}')

# processed_csv = './data/data_processed_DOW_30_TICKER_2009-01-01_to_2022-07-31.csv' 
# processed_csv = './data/nifty/data_processed_nifty_tics_2009_2022_Aug.csv'
processed_csv = './data/sensex/sensex_tics_processed_latest.csv'
print(f'Reading processed csv {processed_csv}')
df_processed = pd.read_csv(processed_csv, index_col='Unnamed: 0')

train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
test = data_split(df_processed, TEST_START_DATE, TEST_END_DATE)
trade = data_split(df_processed, TRADE_START_DATE, TRADE_END_DATE)
trade_2 = data_split(df_processed, TRADE_2_START_DATE, TRADE_2_END_DATE)
    
print('='*100)
print('Train from', train['date'].iloc[0], ' to ', train['date'].iloc[-1])
print('Test from', test['date'].iloc[0], ' to ', test['date'].iloc[-1])
print('Trade from', trade['date'].iloc[0], ' to ', trade['date'].iloc[-1])
print('Trade 2 from', trade_2['date'].iloc[0], ' to ', trade_2['date'].iloc[-1])
print('='*100)

if PERIOD == "monthly":
    train = sample_data_for_every_nth_day_of_the_month(df=train, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    trade = sample_data_for_every_nth_day_of_the_month(df=trade, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    test = sample_data_for_every_nth_day_of_the_month(df=test, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    print('Shapes after converting from daily to monthly')
    print(f'Train shape: {train.shape} Trade shape: {trade.shape} Test shape {test.shape}')

print('='*100)
print('Train from', train['date'].iloc[0], ' to ', train['date'].iloc[-1])
print('Test from', test['date'].iloc[0], ' to ', test['date'].iloc[-1])
print('Trade from', trade['date'].iloc[0], ' to ', trade['date'].iloc[-1])
print('='*100)

# baseline returns for index

baseline_daily_returns_wrt_day_list = get_baseline_daily_returns(baseline_ticker=BASELINE_TICKER_NAME_BACKTESTING,
                                                                train=train)

# print(baseline_daily_returns_wrt_day_list)

# env
stock_dimension = len(train.tic.unique())
state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": HMAX,
    "initial_amount": INITIAL_AMOUNT,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    # "reward_scaling": 1e-4,
    "reward_scaling": 1e-2,
    "baseline_daily_returns": baseline_daily_returns_wrt_day_list,
}
# print('env_kwargs', env_kwargs)
env = StockTradingEnv(df = train, **env_kwargs)
# env = StockTradingEnvCashpenalty(df=train, **env_kwargs)
env_train, _ = env.get_sb_env()

# ENV_NAME = 'stock_tr_scaled_actions'
w_config = dict(
  alpha = ACTOR_LR,
  beta = CRITIC_LR,
  batch_size = BATCH_SIZE,
  layer1_size = LAYER_1_SIZE,
  layer2_size = LAYER_2_SIZE,
  buffer_size = BUFFER_SIZE,
  input_dims = state_space,
  n_actions = stock_dimension,
  tau = TAU,
  architecture = "DDPG",
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
  checkpoint_dir = CHECKPOINT_DIR,
  results_dir = RESULTS_DIR,
  seed = SEED,
  ticker_list_name = BASELINE_TICKER_NAME_BACKTESTING,
  period = PERIOD,
  date_per_month_for_actions = DATE_OF_THE_MONTH_TO_TAKE_ACTIONS,
  EXPLORATION_NOISE = EXPLORATION_NOISE,
  UNIFORM_INITIAL_EXPLORATION = UNIFORM_INITIAL_EXPLORATION,
  EXPLORATION_STEP_COUNT = EXPLORATION_STEP_COUNT,
  NOISE_ANNEALING = NOISE_ANNEALING,
  sigma = sigma, 
  theta = theta,
  dt = dt,
  NORMAL_SCALAR = NORMAL_SCALAR,
  LR_SCHEDULE_STEP_SIZE = LR_SCHEDULE_STEP_SIZE_ACTOR,
  LR_SCHEDULE_STEP_SIZE_CRITIC = LR_SCHEDULE_STEP_SIZE_CRITIC
)

# PROJECT_NAME = f"pytorch_tuned_sb_ddpg_{ENV_NAME.lower()}"
# PROJECT_NAME = "ddpg_tuned_dji_linux"
print('SEED: ', SEED)
np.random.seed(SEED)
env.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# print('Asset: ', env.asset_memory)
# log
noise_params = { # coming from config.py
    'sigma': sigma, 
    'theta': theta, 
    'dt': dt,
}

TOTAL_STEPS_GLOBAL = TOTAL_EPISODES*train.date.nunique()

agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=CHECKPOINT_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=BUFFER_SIZE,
              n_actions=stock_dimension, total_steps_global=TOTAL_STEPS_GLOBAL, **noise_params)

agent = agent.train_model(
                        total_episodes=TOTAL_EPISODES, train_from_scratch=TRAIN_FROM_SCRATCH, 
                        env=env, env_kwargs=env_kwargs, save_ckp=True, ckp_save_freq=SAVE_CKP_AFTER_EVERY_NUM_EPISODES,
                        use_wandb=USE_WANDB, wandb_config=w_config, wandb_project_name=PROJECT_NAME)

print('agent training complete')

# df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=trade, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED)
# results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)

df_account_value_22, df_actions_22, cumulative_rewards_test = trade_on_test_df(df=test, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED, mode='test')
results_df_22 = get_comparison_df(df_account_value_22, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)

df_account_value_2, df_actions_2, cumulative_rewards_test = trade_on_test_df(df=trade_2, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED, mode='test')
results_df_2 = get_comparison_df(df_account_value_2, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)

# saving

account_value_csv_name = f'account_value_test_episode_{agent.episode}.csv'
actions_csv_name = f'daily_actions_test_episode_{agent.episode}.csv'
results_table_name = f'return_comparison_episode_{agent.episode}.csv'
# df_account_value.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name))
# df_actions.to_csv(os.path.join(RESULTS_DIR, actions_csv_name))
# results_df.to_csv(os.path.join(RESULTS_DIR, results_table_name))

df_account_value_22.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name.replace('.csv', '_22.csv')))
df_actions_22.to_csv(os.path.join(RESULTS_DIR, actions_csv_name.replace('.csv', '_22.csv')))
results_df_22.to_csv(os.path.join(RESULTS_DIR, results_table_name.replace('.csv', '_22.csv')))

df_account_value_2.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name.replace('.csv', '_2.csv')))
df_actions_2.to_csv(os.path.join(RESULTS_DIR, actions_csv_name.replace('.csv', '_2.csv')))
results_df_2.to_csv(os.path.join(RESULTS_DIR, results_table_name.replace('.csv', '_2.csv')))

# plotting DJI vs agent cumulative returns
# test_returns_t, baseline_returns_t = backtest_plot(df_account_value, 
#                                                  baseline_ticker = BASELINE_TICKER_NAME_BACKTESTING, 
#                                                  baseline_start = df_account_value.iloc[0]['date'],
#                                                  baseline_end = df_account_value.iloc[-1]['date'])

test_returns_22, baseline_returns_22 = backtest_plot(df_account_value_22, 
                                                 baseline_ticker = BASELINE_TICKER_NAME_BACKTESTING, 
                                                 baseline_start = df_account_value_22.iloc[0]['date'],
                                                 baseline_end = df_account_value_22.iloc[-1]['date'])

test_returns_2, baseline_returns_2 = backtest_plot(df_account_value_2, 
                                                 baseline_ticker = BASELINE_TICKER_NAME_BACKTESTING, 
                                                 baseline_start = df_account_value_2.iloc[0]['date'],
                                                 baseline_end = df_account_value_2.iloc[-1]['date'])


# cum_rets_t = ep.cum_returns(test_returns_t, 0.0)
# cum_rets_dji_t = ep.cum_returns(baseline_returns_t, 0.0)

cum_rets_22 = ep.cum_returns(test_returns_22, 0.0)
cum_rets_dji_22 = ep.cum_returns(baseline_returns_22, 0.0)

cum_rets_2 = ep.cum_returns(test_returns_2, 0.0)
cum_rets_dji_2 = ep.cum_returns(baseline_returns_2, 0.0)

plt.figure(figsize=(16,6))
plt.subplot(211)
plt.plot(cum_rets_2)
plt.plot(cum_rets_dji_2)
plt.legend(['agent', 'index'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns')
plt.savefig(os.path.join(RESULTS_DIR, 'Cumulative returns 1.png'), dpi=600)

plt.subplot(212)
plt.plot(cum_rets_22)
plt.plot(cum_rets_dji_22)
plt.legend(['agent', 'index'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns 22')


# logging
if USE_WANDB:
    # res_table = wandb.Table(dataframe=results_df) 
    res_table_22 = wandb.Table(dataframe=results_df_22)
    res_table_2 = wandb.Table(dataframe=results_df_2)

    # agent.run.log({f'Results {agent.episode}': res_table})
    agent.run.log({f'Results 22 {agent.episode}': res_table_22})
    agent.run.log({f'Results 2 {agent.episode}': res_table_2})
    agent.run.log({"Cumulative returns comparison": plt})

plt.clf()
plt.figure(figsize=(16,6))
# plt.plot(cum_rets_t)
# plt.plot(cum_rets_dji_t)
# plt.legend(['agent', 'dji'])
# plt.xlabel('Date')
# plt.ylabel('Cumulative returns')
# plt.savefig(os.path.join(RESULTS_DIR, 'Cumulative returns Jan 22.png'), dpi=600)

plt.clf()
plt.figure(figsize=(16,6))
plt.plot(cum_rets_22)
plt.plot(cum_rets_dji_22)
plt.legend(['agent', 'dji'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns')
plt.savefig(os.path.join(RESULTS_DIR, 'Cumulative returns May 20.png'), dpi=600)

plt.clf()
plt.figure(figsize=(16,6))
plt.plot(cum_rets_2)
plt.plot(cum_rets_dji_2)
plt.legend(['agent', 'dji'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns')
plt.savefig(os.path.join(RESULTS_DIR, 'Cumulative returns July 22.png'), dpi=600)

# copy config to trained models and results dir
