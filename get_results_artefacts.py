from ddpg_torch import Agent
# from ddpg_torch_3_lay import Agent
import numpy as np
import pandas as pd
import wandb
from config_results import *
from stock_trading_env import StockTradingEnv
# from env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from trade_stocks import trade_on_test_df
import os
from plot import get_comparison_df, backtest_plot
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import matplotlib.pyplot as plt
import empyrical as ep
from utils import sample_data_for_every_nth_day_of_the_month, get_df_cum_ret
import json

if BASELINE_TICKER_NAME_BACKTESTING == '^DJI':
    processed_csv = './data/data_processed_DOW_30_TICKER_2009-01-01_to_2022-07-31.csv'
else:
    processed_csv = './data/sensex/sensex_tics_processed.csv'
print(f'Reading processed csv {processed_csv}')
df_processed = pd.read_csv(processed_csv, index_col='Unnamed: 0')

train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
test = data_split(df_processed, TEST_START_DATE, TEST_END_DATE)

print('='*100)
print('Test from', test['date'].iloc[0], ' to ', test['date'].iloc[-1])
print('='*100)

if PERIOD == "monthly":
    test = sample_data_for_every_nth_day_of_the_month(df=test, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    print('Shapes after converting from daily to monthly')
    print(f'Test shape {test.shape}')

print('='*100)
print('Test from', test['date'].iloc[0], ' to ', test['date'].iloc[-1])
print('='*100)

# env
stock_dimension = len(test.tic.unique())
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
    "reward_scaling": 1e-4
}
print('env_kwargs', env_kwargs)
# env = StockTradingEnv(df = train, **env_kwargs)
# # env = StockTradingEnvCashpenalty(df=train, **env_kwargs)
# env_train, _ = env.get_sb_env()

# # ENV_NAME = 'stock_tr_scaled_actions'
# w_config = dict(
#   alpha = ACTOR_LR,
#   beta = CRITIC_LR,
#   batch_size = BATCH_SIZE,
#   layer1_size = LAYER_1_SIZE,
#   layer2_size = LAYER_2_SIZE,
#   buffer_size = BUFFER_SIZE,
#   input_dims = state_space,
#   n_actions = stock_dimension,
#   tau = TAU,
#   architecture = "DDPG",
#   train_start_date = TRAIN_START_DATE,
#   train_end_date = TRAIN_END_DATE,
#   trade_start_date = TRADE_START_DATE,
#   trade_end_date = TRADE_END_DATE,
#   hmax = env_kwargs['hmax'],
#   initial_amount = env_kwargs['initial_amount'],
#   num_stock_shares = env_kwargs['num_stock_shares'],
#   buy_cost_pct = env_kwargs['buy_cost_pct'],
#   sell_cost_pct = env_kwargs['sell_cost_pct'],
#   state_space = env_kwargs['state_space'],
#   stock_dimension = env_kwargs['stock_dim'],
#   tech_indicator_list = env_kwargs['tech_indicator_list'],
#   action_space = env_kwargs['action_space'],
#   reward_scaling = env_kwargs['reward_scaling'],
#   checkpoint_dir = CHECKPOINT_DIR,
#   results_dir = RESULTS_DIR,
#   seed = SEED,
#   ticker_list_name = BASELINE_TICKER_NAME_BACKTESTING,
#   period = PERIOD,
#   date_per_month_for_actions = DATE_OF_THE_MONTH_TO_TAKE_ACTIONS,
#   EXPLORATION_NOISE = EXPLORATION_NOISE,
#   UNIFORM_INITIAL_EXPLORATION = UNIFORM_INITIAL_EXPLORATION,
#   EXPLORATION_STEP_COUNT = EXPLORATION_STEP_COUNT,
#   NOISE_ANNEALING = NOISE_ANNEALING,
#   sigma = sigma, 
#   theta = theta,
#   dt = dt,
#   NORMAL_SCALAR = NORMAL_SCALAR,
#   LR_SCHEDULE_STEP_SIZE = LR_SCHEDULE_STEP_SIZE_ACTOR,
#   LR_SCHEDULE_STEP_SIZE_CRITIC = LR_SCHEDULE_STEP_SIZE_CRITIC
# )

print('SEED: ', SEED)
np.random.seed(SEED)
# env.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# print('Asset: ', env.asset_memory)
# log

noise_params = { # coming from config.py
    'sigma': sigma, 
    'theta': theta, 
    'dt': dt,
}


agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=CHECKPOINT_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=BUFFER_SIZE,
              n_actions=stock_dimension, **noise_params)

agent.load_checkpoint(checkpoint_path=SAVED_CHECKPOINT_PATH)

print('agent training complete')

# df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=trade, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED)
# results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)

df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=test, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED)
results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)


# saving
account_value_csv_name = f'account_value_test_episode_{agent.episode}.csv'
actions_csv_name = f'daily_actions_test_episode_{agent.episode}.csv'
results_table_name = f'return_comparison_episode_{agent.episode}.csv'
# df_account_value.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name))
# df_actions.to_csv(os.path.join(RESULTS_DIR, actions_csv_name))
# results_df.to_csv(os.path.join(RESULTS_DIR, results_table_name))

df_account_value.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name.replace('.csv', '_22.csv')))
df_actions.to_csv(os.path.join(RESULTS_DIR, actions_csv_name.replace('.csv', '_22.csv')))
results_df.to_csv(os.path.join(RESULTS_DIR, results_table_name.replace('.csv', '_22.csv')))


# plotting index vs agent cumulative returns

test_returns, baseline_returns = backtest_plot(df_account_value, 
                                                 baseline_ticker = BASELINE_TICKER_NAME_BACKTESTING, 
                                                 baseline_start = df_account_value.iloc[0]['date'],
                                                 baseline_end = df_account_value.iloc[-1]['date'])


cum_rets = ep.cum_returns(test_returns, 0.0)
cum_rets_dji = ep.cum_returns(baseline_returns, 0.0)

plt.figure(figsize=(16,6))

plt.plot(cum_rets)
plt.plot(cum_rets_dji)
plt.legend(['agent', 'index'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns')


# logging
if USE_WANDB:
    # res_table = wandb.Table(dataframe=results_df) 
    res_table = wandb.Table(dataframe=results_df)

    # agent.run.log({f'Results {agent.episode}': res_table})
    agent.run.log({f'Results {agent.episode}': res_table})
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
plt.plot(cum_rets)
plt.plot(cum_rets_dji)
plt.legend(['agent', 'dji'])
plt.xlabel('Date')
plt.ylabel('Cumulative returns')
plt.savefig(os.path.join(RESULTS_DIR, 'Cumulative returns May 20.png'), dpi=600)

# json
cum_rets_agent = get_df_cum_ret(cum_rets)
cum_rets_dji_index = get_df_cum_ret(cum_rets_dji)

result = cum_rets_agent.to_json(orient="table")
parsed = json.loads(result) 

result_index = cum_rets_dji_index.to_json(orient="table")
parsed_index = json.loads(result_index)

with open(os.path.join(RESULTS_DIR, 'cumulative_returns_agent.json'), 'w') as f:
    json.dump(parsed, f, indent=4)

with open(os.path.join(RESULTS_DIR, 'cumulative_returns_dji_index.json'), 'w') as f:
    json.dump(parsed_index, f, indent=4)

results_df['metric'][0] = 'Annual return %'
results_df['metric'][1] = 'Cumulative returns %'
results_df['metric'][6] = 'Max drawdown %'
results_df.set_index('metric', inplace=True)
results_df.rename(columns={list(results_df)[0]: 'Agent', list(results_df)[1]: 'DJI'}, inplace=True)

results_df.loc['Annual return %']['Agent'] = results_df.loc['Annual return %']['Agent']*100
results_df.loc['Annual return %']['DJI'] = results_df.loc['Annual return %']['DJI']*100

results_df.loc['Cumulative returns %']['Agent'] = results_df.loc['Cumulative returns %']['Agent']*100
results_df.loc['Cumulative returns %']['DJI'] = results_df.loc['Cumulative returns %']['DJI']*100

results_df.loc['Max drawdown %']['Agent'] = results_df.loc['Max drawdown %']['Agent']*100
results_df.loc['Max drawdown %']['DJI'] = results_df.loc['Max drawdown %']['DJI']*100

results_df = results_df.dropna()
return_comp_df_json = results_df.to_json(orient="table")
parsed_table = json.loads(return_comp_df_json)

with open(os.path.join(RESULTS_DIR, 'return_comparison_table_dji.json'), 'w') as f:
    json.dump(parsed_table, f, indent=4)