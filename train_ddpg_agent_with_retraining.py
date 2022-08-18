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
from trade_stocks import trade_on_test_df, DRL_prediction_retraining
import os
from plot import get_comparison_df, backtest_plot
import warnings
warnings.filterwarnings("ignore")
import random
import torch
import matplotlib.pyplot as plt
import empyrical as ep
from hyperparameter_tuning import get_tuned_hyperparams
# seed fixing for reproducability
np.random.seed(SEED)
# env.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

import config

net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [512, 512],
    }

def run_trade_and_train(retrain_in_months, df, train_period, trade_period, 
                        test_period_in_months=None, do_hyp_tuning=False):
    '''
    retrain_in_months: number of months after which to retrain the agent
    df: entire processed dataframe containing train and trade data
    test_period_in_months: num months before trading to use for hyperparameter tuning
    train_period: train data dates tuple (start_date, end_date)
    trade_period: trade data dates tuple (start_date, end_date)
    do_hyp_tuning: run hyperparameter tuning on test set to find best params for trade set
    '''

    if PERIOD == 'daily':
        retrain_window = 21 * retrain_in_months # avg trading days per month = 21
        test_window = 21 * test_period_in_months

    elif PERIOD == 'monthly':
        retrain_window = retrain_in_months
        test_window = test_period_in_months

    unique_trade_date = df[(df.date > trade_period[0]) & (df.date <= trade_period[1])].date.unique()
    print('Num of unique trade dates: ', len(unique_trade_date))
    print('Retrain window: ', retrain_window)
    print('Testing window: ', test_window)
    
    insample_turbulence = df[(df.date < train_period[1]) & (df.date >= train_period[0])]
    turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.996)
    print("turbulence_threshold: ", turbulence_threshold)

    # fixed env variables for each iteration
    stock_dimension = len(df.tic.unique())
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

    last_state_agent = []
    df_ac_values = []
    for i in range(retrain_window, len(unique_trade_date), retrain_window):
        train_start_date = train_period[0]
        train_end_date = unique_trade_date[i - retrain_window]
        
        train = data_split(df, start=train_start_date, end=train_end_date)
#         print('Train start date:', train['date'].iloc[0], ' Train end date:', train['date'].iloc[-1])

        if test_period_in_months:
            test_end_date = train_end_date
            unique_train_dates = train['date'].unique()
            test_start_date = unique_train_dates[:len(unique_train_dates)-test_window][-1]

            test = data_split(train, start=test_start_date, end=test_end_date)
            train_hyp = data_split(train, start=train_start_date, end=test_start_date)
            
        trade = data_split(df, start=unique_trade_date[i - retrain_window], end=unique_trade_date[i])
        
        print('='*100)
        print('Train start date:', train['date'].iloc[0], ' Train end date:', train['date'].iloc[-1])
        print('Test start date:', test['date'].iloc[0], ' Test end date:', test['date'].iloc[-1])
        print('Trade start date:', trade['date'].iloc[0], ' Trade end date:', trade['date'].iloc[-1])
        
        # hyperparameter tuning on test set
        if do_hyp_tuning:
            hyperparameters = get_tuned_hyperparams(train_df=train_hyp, test_df=test, env_kwargs=env_kwargs, study_name=f'ddpg_study_{i}')
            ACTOR_LR = hyperparameters['learning_rate']
            CRITIC_LR = hyperparameters['learning_rate']
            BATCH_SIZE = hyperparameters['batch_size']
            net_size = net_arch[hyperparameters['net_arch']]
            LAYER_1_SIZE = net_size[0]
            LAYER_2_SIZE = net_size[1]
            # BUFFER_SIZE = hyperparameters['buffer_size']

        else:
            ACTOR_LR = config.ACTOR_LR
            CRITIC_LR = config.CRITIC_LR
            BATCH_SIZE = config.BATCH_SIZE
            LAYER_1_SIZE = config.LAYER_1_SIZE
            LAYER_2_SIZE = config.LAYER_2_SIZE
            # BUFFER_SIZE = hyperparameters['buffer_size']

        train_env = StockTradingEnv(df=train, **env_kwargs)
        train_env.seed(SEED)

        if i - retrain_window == 0:
            # inital state
            initial = True
        else:
            # previous state
            initial = False

        print(
                "======Training from: ",
                list(train['date'])[0],
                "to ",
                list(train['date'])[-1],
            )
        
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
                train_csv = TRAIN_CSV_NAME,
                trade_csv = TRADE_CSV_NAME,
                seed = SEED,
                ticker_list_name = ticker_name_from_config_tickers,
                period = PERIOD,
                date_per_month_for_actions = DATE_OF_THE_MONTH_TO_TAKE_ACTIONS,
                retrain_in_months = RETRAIN_IN_MONTHS,
                test_in_months = TEST_PERIOD_IN_MONTHS
                )
                    
        agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=CHECKPOINT_DIR, input_dims=state_space, tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=BUFFER_SIZE,
              n_actions=stock_dimension)
        
        agent = agent.train_model(
                        total_episodes=TOTAL_EPISODES, train_from_scratch=TRAIN_FROM_SCRATCH, 
                        env=train_env, env_kwargs=env_kwargs, save_ckp=True, ckp_save_freq=SAVE_CKP_AFTER_EVERY_NUM_EPISODES,
                        use_wandb=USE_WANDB, wandb_config=w_config, wandb_project_name=PROJECT_NAME)

        print(
                "======Trading from: ",
                list(trade['date'])[0],
                "to ",
                list(trade['date'])[-1],
            )

        df_account_value, df_actions, last_state_agent = DRL_prediction_retraining(
                                                                                    trade_data=trade, 
                                                                                    model=agent,
                                                                                    iter_num=i, 
                                                                                    last_state=last_state_agent, 
                                                                                    turbulence_threshold=turbulence_threshold, 
                                                                                    initial=initial, 
                                                                                    env_kwargs=env_kwargs, 
                                                                                    seed=SEED)
        
        
        results_df = get_comparison_df(df_account_value, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)
        df_ac_values.append(df_account_value)

        account_value_csv_name = f'account_value_{i}.csv'
        actions_csv_name = f'daily_actions_test_{i}.csv'
        results_table_name = f'return_comparison_episode_{i}.csv'

        df_account_value.to_csv(os.path.join(RESULTS_DIR, account_value_csv_name))
        df_actions.to_csv(os.path.join(RESULTS_DIR, actions_csv_name))
        results_df.to_csv(os.path.join(RESULTS_DIR, results_table_name))
        
        if USE_WANDB:
            res_table = wandb.Table(dataframe=results_df) 
            agent.run.log({f'Results for {i}': res_table})

        
    df_final_account_value_full_trade_window = pd.concat(df_ac_values)
    idx, i = [], 0
    cur_date = df_final_account_value_full_trade_window.date.iloc[0]
    for j in range(len(df_final_account_value_full_trade_window)):
        if df_final_account_value_full_trade_window.date.iloc[j] == cur_date:
            idx.append(i)
        else:
            cur_date = df_final_account_value_full_trade_window.date.iloc[j] 
            i += 1
            idx.append(i)

    df_final_account_value_full_trade_window = df_final_account_value_full_trade_window.set_axis(idx)
    df_final_account_value_full_trade_window.to_csv(os.path.join(RESULTS_DIR, 'final_account_value.csv'))
    
    results_df = get_comparison_df(df_final_account_value_full_trade_window, BASELINE_TICKER_NAME_BACKTESTING, period=PERIOD)
    test_returns_t, baseline_returns_t = backtest_plot(df_final_account_value_full_trade_window, 
                                                 baseline_ticker = BASELINE_TICKER_NAME_BACKTESTING, 
                                                 baseline_start = df_final_account_value_full_trade_window.iloc[0]['date'],
                                                 baseline_end = df_final_account_value_full_trade_window.iloc[-1]['date'])


    cum_rets_t = ep.cum_returns(test_returns_t, 0.0)
    cum_rets_dji_t = ep.cum_returns(baseline_returns_t, 0.0)

    plt.figure(figsize=(16,6))
    plt.plot(cum_rets_t)
    plt.plot(cum_rets_dji_t)
    plt.legend(['agent', 'dji'])
    plt.xlabel('Date')
    plt.ylabel('Cumulative returns')

    # logging
    if USE_WANDB:
        res_table = wandb.Table(dataframe=results_df) 
        agent.run.log({f'Results Episode {agent.episode}': res_table})

        agent.run.log({"Cumulative returns comparison": plt})

    print('DONE')
    
if __name__ == "__main__":
    # if not os.path.exists(PROCESSED_CSV_NAME):
    #     df_raw = YahooDownloader(start_date = TRAIN_START_DATE,
    #                     end_date = TRADE_END_DATE,
    #                     ticker_list = config_tickers.DOW_30_TICKER).fetch_data()
    #     print(f'Data Downloaded! shape:{df_raw.shape}')

    #     df = process_data(df_raw, use_technical_indicator=True, technical_indicator_list=INDICATORS, 
    #                                 use_vix=True, use_turbulence=True, user_defined_feature=False)

    #     print(f'Data Processed! shape:{df.shape}')
    # else:
    #     print(f'\nreading csv {PROCESSED_CSV_NAME}...')
    #     df = pd.read_csv(PROCESSED_CSV_NAME, index_col='Unnamed: 0')
    #     print(f'DF shape: {df.shape}')
    processed_csv = './data/data_processed_DOW_30_TICKER_2009-01-01_to_2022-07-31.csv'
    
    print(f'Reading processed csv {processed_csv}')
    df_processed = pd.read_csv(processed_csv, index_col='Unnamed: 0')

    df = data_split(df_processed, TRAIN_START_DATE, TRADE_END_DATE)

    print('DF shape: ', df.shape)
    print('DF data from', df['date'].iloc[0], ' to ', df['date'].iloc[-1])
    # if any([dt for dt in df.date.unique() if dt < TRAIN_START_DATE]):
    #     df = df[df.date >= TRAIN_START_DATE]
    #     idx, i = [], 0
    #     cur_date = df.date.iloc[0]
    #     for j in range(len(df)):
    #         if df.date.iloc[j] == cur_date:
    #             idx.append(i)
    #         else:
    #             cur_date = df.date.iloc[j] 
    #             i += 1
    #             idx.append(i)

    #     df = df.set_axis(idx)

    if PERIOD == "monthly":
        df = sample_data_for_every_nth_day_of_the_month(df=df, date=DATE_OF_THE_MONTH_TO_TAKE_ACTIONS)
    
        print('Shape after converting from daily to monthly')
        print(f'DF shape: {df.shape}')
    
    run_trade_and_train(
                        retrain_in_months=RETRAIN_IN_MONTHS,
                        df=df, 
                        train_period=(TRAIN_START_DATE, TRAIN_END_DATE), 
                        trade_period=(TRADE_START_DATE, TRADE_END_DATE),
                        test_period_in_months=TEST_PERIOD_IN_MONTHS,
                        do_hyp_tuning=True
                        )