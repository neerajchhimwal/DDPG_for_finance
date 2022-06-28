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

import os

import warnings
warnings.filterwarnings("ignore")

ENV_NAME = 'stock_trading'

# DATA

# df = download(TRAIN_START_DATE, VALID_END_DATE, ticker_list=config_tickers.DOW_30_TICKER)
# if not os.path.exists(TRAIN_CSV_NAME):
df = YahooDownloader(start_date = TRAIN_START_DATE,
                    end_date = VALID_END_DATE,
                    ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
print(f'Data Downloaded! shape:{df.shape}')

df_processed = process_data(df, use_technical_indicator=True, technical_indicator_list=INDICATORS, 
                            use_vix=True, use_turbulence=True, user_defined_feature=False)

print(f'Data Processed! shape:{df_processed.shape}')

train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(df_processed, VALID_START_DATE, VALID_END_DATE)

print(f'Train shape: {train.shape} Trade shape: {trade.shape}')
print('Saving csvs...')
df.to_csv(ORIGINAL_CSV_NAME)
df_processed.to_csv(PROCESSED_CSV_NAME)
train.to_csv(TRAIN_CSV_NAME)
trade.to_csv(TRADE_CSV_NAME)
# else:
#     train = pd.read_csv(TRAIN_CSV_NAME)
#     trade = pd.read_csv(TRADE_CSV_NAME)
#     print(f'Train shape: {train.shape}\nTrade shape: {trade.shape}')

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

# log
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
  trade_start_date = VALID_START_DATE,
  trade_end_date = VALID_END_DATE,
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

PROJECT_NAME = f"pytorch_ddpg_{ENV_NAME.lower()}"

if USE_WANDB:
    run = wandb.init(project=PROJECT_NAME, tags=["DDPG", "RL"], config=w_config, job_type='train_model')



agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, input_dims=state_space, tau=TAU, env=ENV_NAME,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=50000,
              n_actions=stock_dimension)

starting_episode = 0
if not TRAIN_FROM_SCRATCH:
    agent.load_models()
    starting_episode = agent.episode + 1
 
np.random.seed(SEED)

score_history = []
for i in range(starting_episode, TOTAL_EPISODES):
    obs = env.reset()
    done = False
    score = 0
    step_count = 0
    actor_loss_per_step_list = []
    critic_loss_per_step_list = []
    agent.episode = i
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        step_count += 1 
        # run.log({'Actor Loss - STEP': agent.actor_loss_step})
        # run.log({'Critic Loss - STEP': agent.critic_loss_step}) 
        actor_loss_per_step_list.append(agent.actor_loss_step)
        critic_loss_per_step_list.append(agent.critic_loss_step)
        if step_count % 50 == 0:
            print(f'actor loss step {step_count}: {agent.actor_loss_step}')
            print(f'critic loss step {step_count}: {agent.critic_loss_step}')
        # this is a temporal diff method: we learn at each timestep, 
        # unline monte carlo methods where learning is done at the end of an episode
        score += reward
        obs = new_state
        # env.render()
    score_history.append(score)
    if USE_WANDB:
        run.log({'steps per episode': step_count, 'episode': i})
        run.log({'reward': score, 'episode': i})
        # run.log({'reward avg 100 games': np.mean(score_history[-100:]), 'episode': i})
        run.log({'Actor loss': np.mean(actor_loss_per_step_list), 'episode': i})
        run.log({'Critic loss': np.mean(critic_loss_per_step_list), 'episode': i})

    if i % SAVE_CKP_AFTER_EVERY_NUM_EPISODES == 0:
       agent.save_models()
    print('='*50)
    print('episode ', i, 'reward %.2f' % score)
    print('='*50)

filename = f'{ENV_NAME}-alpha{str(ACTOR_LR)}-beta{str(CRITIC_LR)}batch_{BATCH_SIZE}-400-300.png'
plotLearning(score_history, filename, window=1)