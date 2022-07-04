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

import warnings
warnings.filterwarnings("ignore")

ENV_NAME = 'stock_trading'

# DATA

# df = download(TRAIN_START_DATE, TRADE_END_DATE, ticker_list=config_tickers.DOW_30_TICKER)
if not (os.path.exists(TRAIN_CSV_NAME) or os.path.exists(TEST_CSV_NAME) or os.path.exists(TRADE_CSV_NAME)):
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
# print('Asset: ', env.asset_memory)
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

PROJECT_NAME = f"pytorch_ddpg_{ENV_NAME.lower()}"

if USE_WANDB:
    run = wandb.init(project=PROJECT_NAME, tags=["DDPG", "RL"], config=w_config, job_type='train_model') #, resume=RESUME_LAST_WANDB_RUN)



agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, input_dims=state_space, tau=TAU, env=ENV_NAME,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=50000,
              n_actions=stock_dimension)

starting_episode = 0
if not TRAIN_FROM_SCRATCH:
    agent.load_models()
    starting_episode = agent.episode + 1
 
np.random.seed(SEED)

day_counter_total = 0
score_history = []
for i in range(starting_episode, TOTAL_EPISODES):
    obs = env.reset()
    done = False
    score = 0
    step_count = 0
    actor_loss_per_episode = 0
    critic_loss_per_episode = 0
    agent.episode = i
    cumulative_rewards_per_step_this_episode = []
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        step_count += 1
        cumulative_reward = (env.asset_memory[-1] - env_kwargs['initial_amount']) / env_kwargs['initial_amount']
        cumulative_rewards_per_step_this_episode.append(cumulative_reward)
        if USE_WANDB:
            run.log({'Cumulative returns': cumulative_reward, 'days':day_counter_total})
        day_counter_total += 1
        # run.log({'Actor Loss - STEP': agent.actor_loss_step})
        # run.log({'Critic Loss - STEP': agent.critic_loss_step}) 
        actor_loss_per_episode += agent.actor_loss_step
        critic_loss_per_episode += agent.critic_loss_step
        # if step_count % 50 == 0:
        #     print(f'actor loss step {step_count}: {agent.actor_loss_step}')
        #     print(f'critic loss step {step_count}: {agent.critic_loss_step}')
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
        run.log({'Actor loss': actor_loss_per_episode, 'episode': i})
        run.log({'Critic loss': critic_loss_per_episode, 'episode': i})

    
    if i % SAVE_CKP_AFTER_EVERY_NUM_EPISODES == 0 or i==TOTAL_EPISODES-1:
       agent.save_models()
    # print('='*50)
    # print('episode ', i, 'reward %.2f' % score)
    # print('='*50)

    # test cumulative returns on test set
    


    if i % SAVE_REWARD_TABLE_AFTER_EVERY_NUM_EPISODES == 0 or i==TOTAL_EPISODES-1:
        df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=test, model=agent, train_df=train, env_kwargs=env_kwargs)
        print('results table....')
        print(df_account_value.head())
        results_dir = './results'
        account_value_csv_name = f'account_value_test_episode_{i}.csv'
        actions_csv_name = f'daily_actions_test_episode_{i}.csv'
        results_table_name = f'return_comparison_episode_{i}.csv'
        df_account_value.to_csv(os.path.join(results_dir, account_value_csv_name))
        df_actions.to_csv(os.path.join(results_dir, actions_csv_name))

        df = pd.DataFrame(data=[cumulative_rewards_test[-1], max(cumulative_rewards_test), min(cumulative_rewards_test)],
                        columns=[f'test [{TEST_START_DATE}_{TEST_END_DATE}]'],
                        index=['Cumulative Return', 'Max Cumulative return', 'Min cumulative return'])

        df[f'train [{TRAIN_START_DATE}_{TRAIN_END_DATE}]'] = [cumulative_rewards_per_step_this_episode[-1], 
                                                            max(cumulative_rewards_per_step_this_episode),
                                                            min(cumulative_rewards_per_step_this_episode)]
        df.to_csv(os.path.join(results_dir, results_table_name))

        if USE_WANDB:
            df.reset_index(inplace=True)
            res_table = wandb.Table(dataframe=df) 
            run.log({f'Cumulative returns Episode {i}': res_table})
            
    



# TRADING
'''
insample_risk_indicator = train.drop_duplicates(subset=['date'])
turb_threshold = insample_risk_indicator.turbulence.quantile(0.996)
turb_threshold = int(round(turb_threshold))
print('='*100)
print('='*100)
print(f'Using Turbulence threshld: {turb_threshold}')
print('='*100)
print('='*100)

# trade_env = StockTradingEnv(df=trade, turbulence_threshold=turb_threshold, risk_indicator_col='turbulence', **env_kwargs)

# df_account_value, df_actions = DRLAgent.DRL_prediction(model=agent, environment=trade_env)
'''