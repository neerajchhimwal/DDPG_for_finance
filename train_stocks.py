from ddpg_torch import Agent
import gym
import numpy as np
import pandas as pd
from utils import sample_data_for_every_nth_day_of_the_month
import wandb
from config import *
import config_tickers
from download_data import process_data
from stock_trading_env import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import data_split
from trade_stocks import trade_on_test_df
import os
from plot import get_comparison_df
import warnings
warnings.filterwarnings("ignore")
import random
import torch

ENV_NAME = 'stock_tr_scaled_actions'

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

if PERIOD == "monthly":
    train = sample_data_for_every_nth_day_of_the_month(df=train, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    trade = sample_data_for_every_nth_day_of_the_month(df=trade, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    test = sample_data_for_every_nth_day_of_the_month(df=test, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    print('Shapes after converting from daily to monthly')
    print(f'Train shape: {train.shape} Trade shape: {trade.shape} Test shape {test.shape}')

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

np.random.seed(SEED)
env.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
# print('Asset: ', env.asset_memory)
# log
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
  ticker_list_name = ticker_name_from_config_tickers,
  period = PERIOD,
  date_per_month_for_actions = DATE_OF_THE_MONTH_TO_TAKE_ACTIONS
)

# PROJECT_NAME = f"pytorch_tuned_sb_ddpg_{ENV_NAME.lower()}"
PROJECT_NAME = "ddpg_tuned_dji_linux"

if USE_WANDB:
    run = wandb.init(project=PROJECT_NAME, tags=["DDPG", "RL"], config=w_config, job_type='train_model') #, resume=RESUME_LAST_WANDB_RUN)



# agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, input_dims=state_space, tau=TAU, env=ENV_NAME,
#               batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=100000,
#               n_actions=stock_dimension)

agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=CHECKPOINT_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=BUFFER_SIZE,
              n_actions=stock_dimension)

starting_episode = 0
if not TRAIN_FROM_SCRATCH:
    starting_episode = agent.load_checkpoint()
 
day_counter_total = 0
score_history = []
for i in range(starting_episode, TOTAL_EPISODES):
    obs = env.reset()
    agent.noise.reset()
    done = False
    score = 0
    step_count = 0
    actor_loss_per_episode = 0
    critic_loss_per_episode = 0
    # actor_loss_per_episode = []
    # critic_loss_per_episode = []
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
        
        actor_loss_per_episode += agent.actor_loss_step
        critic_loss_per_episode += agent.critic_loss_step
        # actor_loss_per_episode.append(agent.actor_loss_step)
        # critic_loss_per_episode.append(agent.critic_loss_step)

        # if step_count % 50 == 0:
        #     print(f'actor loss step {step_count}: {agent.actor_loss_step}')
        #     print(f'critic loss step {step_count}: {agent.critic_loss_step}')
        # this is a temporal diff method: we learn at each timestep, 
        # unline monte carlo methods where learning is done at the end of an episode
        score += reward
        obs = new_state
        # env.render()

    # actor_loss_per_episode = np.mean(actor_loss_per_episode)
    # critic_loss_per_episode = np.mean(critic_loss_per_episode)
    print('$'*100)
    cr_lr = agent.critic.optimizer.state_dict()['param_groups'][0]['lr']
    ac_lr = agent.actor.optimizer.state_dict()['param_groups'][0]['lr']
    print(f"Episode {i}, critic LR: {cr_lr}, critic loss: {critic_loss_per_episode}")
    print(f"Episode {i},  actor LR: {ac_lr},   actor loss: {actor_loss_per_episode}")
    print('$'*100)
    # agent.critic.scheduler.step(critic_loss_per_episode)
    agent.critic.scheduler.step()
    agent.actor.scheduler.step()

    score_history.append(score)
    if USE_WANDB:
        run.log({'steps per episode': step_count, 'episode': i})
        run.log({'reward': score, 'episode': i})
        # run.log({'reward avg 100 games': np.mean(score_history[-100:]), 'episode': i})
        run.log({'Actor loss': actor_loss_per_episode, 'episode': i})
        run.log({'Critic loss': critic_loss_per_episode, 'episode': i})

        run.log({'Actor LR': ac_lr, 'episode': i})
        run.log({'Critic LR': cr_lr, 'episode': i})

    
    if i % SAVE_CKP_AFTER_EVERY_NUM_EPISODES == 0 or i==TOTAL_EPISODES-1:
        # agent.save_models()
        agent.save_checkpoint(last_episode=i)
    
    # test cumulative returns on test set

    if i == TOTAL_EPISODES-1:
        df_account_value, df_actions, cumulative_rewards_test = trade_on_test_df(df=trade, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED)
        # print('results table....')
        # print(df_account_value.head())
        results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)
        train_values = np.zeros(len(results_df))
        train_values[list(results_df.metric).index('Cumulative returns')] = cumulative_rewards_per_step_this_episode[-1]
        train_values[list(results_df.metric).index('Max drawdown')] = min(cumulative_rewards_per_step_this_episode)
        results_df['train_data'] = train_values

        # df_account_value_22, df_actions_22, cumulative_rewards_test_22 = trade_on_test_df(df=test, model=agent, train_df=train, env_kwargs=env_kwargs, seed=SEED)
        # results_df_22 = get_comparison_df(df_account_value_22, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)
        
        # results_df_22['train_data'] = train_values

        # saving
        results_dir = f'./results/monthly_data_seed_{SEED}_trade_from_2020_july'
        # results_dir = './results_lr_schedule_step_10_grad_clip_small_nw_400_400_2016_2022_may'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        account_value_csv_name = f'account_value_test_episode_{i}.csv'
        actions_csv_name = f'daily_actions_test_episode_{i}.csv'
        results_table_name = f'return_comparison_episode_{i}.csv'
        df_account_value.to_csv(os.path.join(results_dir, account_value_csv_name))
        df_actions.to_csv(os.path.join(results_dir, actions_csv_name))
        results_df.to_csv(os.path.join(results_dir, results_table_name))

        # df_account_value_22.to_csv(os.path.join(results_dir, account_value_csv_name.replace('.csv', '_22.csv')))
        # df_actions_22.to_csv(os.path.join(results_dir, actions_csv_name.replace('.csv', '_22.csv')))
        # results_df_22.to_csv(os.path.join(results_dir, results_table_name.replace('.csv', '_22.csv')))

        # logging
        if USE_WANDB:
            res_table = wandb.Table(dataframe=results_df) 
            run.log({f'Results Episode {i}': res_table})

            # res_table_22 = wandb.Table(dataframe=results_df_22)
            # run.log({f'Results 22 Episode {i}': res_table_22})

            