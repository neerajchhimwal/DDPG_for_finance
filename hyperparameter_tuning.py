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

# from config_tuning import *
import config_tuning
from config import SEED, PERIOD, HMAX, INDICATORS, DATE_OF_THE_MONTH_TO_TAKE_ACTIONS, BASELINE_TICKER_NAME_BACKTESTING
# from hyp_utils import *

## Fixed
tpm_hist = {}  # record tp metric values for trials
tp_metric = 'avgwl'  # specified trade_param_metric: ratio avg value win/loss
## Settable by User
n_trials = config_tuning.N_TRIALS  # number of HP optimization runs
total_episodes = config_tuning.TOTAL_EPISODES # per HP optimization run
## Logging callback params
lc_threshold=1e-5
lc_patience=10
lc_trial_number=config_tuning.N_TRIALS

#Main method
# Calculates Trade Performance for Objective
# Called from objective method
# Returns selected trade perf metric(s)
# Requires actions and associated prices

def calc_trade_perf_metric(df_actions, 
                           df_prices_trade,
                           tp_metric,
                           tpm_hist,
                           dbg=False):
  
    df_actions_p, df_prices_p, tics = prep_data(df_actions.copy(),
                                                df_prices_trade.copy())
    # actions predicted by trained model on trade data
    # df_actions_p.to_csv('df_actions.csv') 

    
    # Confirms that actions, prices and tics are consistent
    df_actions_s, df_prices_s, tics_prtfl = \
        sync_tickers(df_actions_p.copy(),df_prices_p.copy(),tics)
    
    # copy to ensure that tics from portfolio remains unchanged
    tics = tics_prtfl.copy()
    
    # Analysis is performed on each portfolio ticker
    perf_data= collect_performance_data(df_actions_s, df_prices_s, tics)
    # profit/loss for each ticker
    pnl_all = calc_pnl_all(perf_data, tics)
    # values for trade performance metrics
    perf_results = calc_trade_perf(pnl_all)
    df = pd.DataFrame.from_dict(perf_results, orient='index')
    
    # calculate and return trade metric value as objective
    m = calc_trade_metric(df,tp_metric)
    print(f'Ratio Avg Win/Avg Loss: {m}')
    k = str(len(tpm_hist)+1)
    # save metric value
    tpm_hist[k] = m
    return m

# Supporting methods
def calc_trade_metric(df,metric='avgwl'):
    '''# trades', '# wins', '# losses', 'wins total value', 'wins avg value',
       'losses total value', 'losses avg value'''
    # For this tutorial, the only metric available is the ratio of 
    #  average values of winning to losing trades. Others are in development.
    
    # some test cases produce no losing trades.
    # The code below assigns a value as a multiple of the highest value during
    # previous hp optimization runs. If the first run experiences no losses,
    # a fixed value is assigned for the ratio
    tpm_mult = 1.0
    avgwl_no_losses = 25
    if metric == 'avgwl':
        if sum(df['# losses']) == 0:
            try:
                return max(tpm_hist.values())*tpm_mult
            except ValueError:
                return avgwl_no_losses
        avg_w = sum(df['wins total value'])/sum(df['# wins'])
        avg_l = sum(df['losses total value'])/sum(df['# losses'])
        m = abs(avg_w/avg_l)

    return m


def prep_data(df_actions,
              df_prices_trade):
    
    df=df_prices_trade[['date','close','tic']]
    df['Date'] = pd.to_datetime(df['date'])
    df = df.set_index('Date')
    # set indices on both df to datetime
    idx = pd.to_datetime(df_actions.index, infer_datetime_format=True)
    df_actions.index=idx
    tics = np.unique(df.tic)
    n_tics = len(tics)
    print(f'Number of tickers: {n_tics}')
    print(f'Tickers: {tics}')
    dategr = df.groupby('tic')
    p_d={t:dategr.get_group(t).loc[:,'close'] for t in tics}
    df_prices = pd.DataFrame.from_dict(p_d)
    df_prices.index = df_prices.index.normalize()
    return df_actions, df_prices, tics


# prepares for integrating action and price files
def link_prices_actions(df_a,
                        df_p):
    cols_a = [t + '_a' for t in df_a.columns]
    df_a.columns = cols_a
    cols_p = [t + '_p' for t in df_p.columns]
    df_p.columns = cols_p
    return df_a, df_p


def sync_tickers(df_actions,df_tickers_p,tickers):
    # Some DOW30 components may not be included in portfolio
    # passed tickers includes all DOW30 components
    # actions and ticker files may have different length indices
    if len(df_actions) != len(df_tickers_p):
        msng_dates = set(df_actions.index)^set(df_tickers_p.index)
        try:
            #assumption is prices has one additional timestamp (row)
            df_tickers_p.drop(msng_dates,inplace=True)
        except:
            df_actions.drop(msng_dates,inplace=True)
    df_actions, df_tickers_p = link_prices_actions(df_actions,df_tickers_p)
    # identify any DOW components not in portfolio
    t_not_in_a = [t for t in tickers if t + '_a' not in list(df_actions.columns)]
  
    # remove t_not_in_a from df_tickers_p
    drop_cols = [t + '_p' for t in t_not_in_a]
    df_tickers_p.drop(columns=drop_cols,inplace=True)
    
    # Tickers in portfolio
    tickers_prtfl = [c.split('_')[0] for c in df_actions.columns]
    return df_actions,df_tickers_p, tickers_prtfl

def collect_performance_data(dfa,dfp,tics, dbg=False):
    
    perf_data = {}
    # In current version, files columns include secondary identifier
    for t in tics:
        # actions: purchase/sale of DOW equities
        acts = dfa['_'.join([t,'a'])].values
        # ticker prices
        prices = dfp['_'.join([t,'p'])].values
        # market value of purchases/sales
        tvals_init = np.multiply(acts,prices)
        d={'actions':acts, 'prices':prices,'init_values':tvals_init}
        perf_data[t]=d

    return perf_data


def calc_pnl_all(perf_dict, tics_all):
    # calculate profit/loss for each ticker
    print(f'Calculating profit/loss for each ticker')
    pnl_all = {}
    for tic in tics_all:
        pnl_t = []
        tic_data = perf_dict[tic]
        init_values = tic_data['init_values']
        acts = tic_data['actions']
        prices = tic_data['prices']
        cs = np.cumsum(acts)
        args_s = [i + 1 for i in range(len(cs) - 1) if cs[i + 1] < cs[i]]
        # tic actions with no sales
        if not args_s:
            pnl = complete_calc_buyonly(acts, prices, init_values)
            pnl_all[tic] = pnl
            continue
        # copy acts: acts_rev will be revised based on closing/reducing init positions
        pnl_all = execute_position_sales(tic,acts,prices,args_s,pnl_all)

    return pnl_all


def complete_calc_buyonly(actions, prices, init_values):
    # calculate final pnl for each ticker assuming no sales
    fnl_price = prices[-1]
    final_values = np.multiply(fnl_price, actions)
    pnl = np.subtract(final_values, init_values)
    return pnl


def execute_position_sales(tic,acts,prices,args_s,pnl_all):
  # calculate final pnl for each ticker with sales
    pnl_t = []
    acts_rev = acts.copy()
    # location of sales transactions
    for s in args_s:  # s is scaler
        # price_s = [prices[s]]
        act_s = [acts_rev[s]]
        args_b = [i for i in range(s) if acts_rev[i] > 0]
        prcs_init_trades = prices[args_b]
        acts_init_trades = acts_rev[args_b]
  
        # update actions for sales
        # reduce/eliminate init values through trades
        # always start with earliest purchase that has not been closed through sale
        # selectors for purchase and sales trades
        # find earliest remaining purchase
        arg_sel = min(args_b)
        # sel_s = len(acts_trades) - 1

        # closing part/all of earliest init trade not yet closed
        # sales actions are negative
        # in this test case, abs_val of init and sales share counts are same
        # zero-out sales actions
        # market value of sale
        # max number of shares to be closed: may be less than # originally purchased
        acts_shares = min(abs(act_s.pop()), acts_rev[arg_sel])

        # mv of shares when purchased
        mv_p = abs(acts_shares * prices[arg_sel])
        # mv of sold shares
        mv_s = abs(acts_shares * prices[s])

        # calc pnl
        pnl = mv_s - mv_p
        # reduce init share count
        # close all/part of init purchase
        acts_rev[arg_sel] -= acts_shares
        acts_rev[s] += acts_shares
        # calculate pnl for trade
        # value of associated purchase
        
        # find earliest non-zero positive act in acts_revs
        pnl_t.append(pnl)
    
    pnl_op = calc_pnl_for_open_positions(acts_rev, prices)
    #pnl_op is list
    # add pnl_op results (if any) to pnl_t (both lists)
    pnl_t.extend(pnl_op)
    #print(f'Total pnl for {tic}: {np.sum(pnl_t)}')
    pnl_all[tic] = np.array(pnl_t)
    return pnl_all


def calc_pnl_for_open_positions(acts,prices):
    # identify any positive share values after accounting for sales
    pnl = []
    fp = prices[-1] # last price
    open_pos_arg = np.argwhere(acts>0)
    if len(open_pos_arg)==0:
        return pnl # no open positions

    mkt_vals_open = np.multiply(acts[open_pos_arg], prices[open_pos_arg])
    # mkt val at end of testing period
    # treat as trades for purposes of calculating pnl at end of testing period
    mkt_vals_final = np.multiply(fp, acts[open_pos_arg])
    pnl_a = np.subtract(mkt_vals_final, mkt_vals_open)
    #convert to list
    pnl = [i[0] for i in pnl_a.tolist()]
    #print(f'Market value of open positions at end of testing {pnl}')
    return pnl


def calc_trade_perf(pnl_d):
    # calculate trade performance metrics
    perf_results = {}
    for t,pnl in pnl_d.items():
        wins = pnl[pnl>0]  # total val
        losses = pnl[pnl<0]
        n_wins = len(wins)
        n_losses = len(losses)
        n_trades = n_wins + n_losses
        wins_val = np.sum(wins)
        losses_val = np.sum(losses)
        wins_avg = 0 if n_wins==0 else np.mean(wins)
        #print(f'{t} n_wins: {n_wins} n_losses: {n_losses}')
        losses_avg = 0 if n_losses==0 else np.mean(losses)
        d = {'# trades':n_trades,'# wins':n_wins,'# losses':n_losses,
             'wins total value':wins_val, 'wins avg value':wins_avg,
             'losses total value':losses_val, 'losses avg value':losses_avg,}
        perf_results[t] = d
    return perf_results

class LoggingCallback:
    def __init__(self, threshold, trial_number, patience):
        '''
        threshold:int tolerance for increase in objective
        trial_number: int Prune after minimum number of trials
        patience: int patience for the threshold
        '''
        self.threshold = threshold
        self.trial_number  = trial_number
        self.patience = patience
        print(f'Callback threshold {self.threshold}, \
            trial_number {self.trial_number}, \
            patience {self.patience}')
        self.cb_list = [] #Trials list for which threshold is reached
        
    def __call__(self, study:optuna.study, frozen_trial:optuna.Trial):
        #Setting the best value in the current trial
        study.set_user_attr("previous_best_value", study.best_value)

        #Checking if the minimum number of trials have pass
        if frozen_trial.number >self.trial_number:
            previous_best_value = study.user_attrs.get("previous_best_value",None)
            #Checking if the previous and current objective values have the same sign
            if previous_best_value * study.best_value >=0:
                #Checking for the threshold condition
                if abs(previous_best_value-study.best_value) < self.threshold: 
                    self.cb_list.append(frozen_trial.number)
                    #If threshold is achieved for the patience amount of time
                    if len(self.cb_list)>self.patience:
                        print('The study stops now...')
                        print('With number',frozen_trial.number ,'and value ',frozen_trial.value)
                        print('The previous and current best values are {} and {} respectively'
                              .format(previous_best_value, study.best_value))
                        study.stop()

def sample_ddpg_params(trial:optuna.Trial):
    # Size of the replay buffer
    # buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # tau = trial.suggest_categorical("tau", [0.01, 0.001, 0.005])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
#     learning_rate_critic = trial.suggest_loguniform("learning_rate_critic", 1e-5, 1e-1)
#     learning_rate_actor = 10**trial.suggest_int('logval', -5, 0)
#     learning_rate_critic = 10**trial.suggest_int('logval', -5, 0)
    if PERIOD=='monthly':
        batch_size = trial.suggest_categorical("batch_size", [4, 8, 16, 32, 64])
    else:
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])

    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "default": [400, 400],
        "big": [512, 512],
    }[net_arch]

    return {
            # "tau": tau,
            # "buffer_size": buffer_size,
            "learning_rate": learning_rate,
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
    buffer_size = 10000
    # TAU = hyperparameters["tau"]
    LAYER_1_SIZE = hyperparameters["layer_1_size"]
    LAYER_2_SIZE = hyperparameters["layer_2_size"]
    # buffer_size = hyperparameters["buffer_size"]
    
    model_ddpg = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, ckp_dir=config_tuning.TUNING_TRIAL_MODELS_DIR, input_dims=env_kwargs['state_space'], tau=TAU,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, max_size=buffer_size,
              n_actions=env_kwargs['stock_dim'])
    
    trained_ddpg = model_ddpg.train_model(
                        total_episodes=total_episodes, train_from_scratch=True, 
                        env=e_train_gym, env_kwargs=env_kwargs, save_ckp=False, ckp_save_freq=0,
                        use_wandb=False)
    
    if config_tuning.SAVE_MODELS:
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

def get_tuned_hyperparams(train_df, test_df, env_kwargs, study_name='ddpg_study'):

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs, print_verbosity=n_trials)
    env_train, _ = e_train_gym.get_sb_env()

    # seed fixing for reproducability
    print(f'SEED: {SEED}')
    np.random.seed(SEED)
    e_train_gym.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    global tpm_hist
    tpm_hist = {}

    if config_tuning.SAVE_MODELS:
        os.makedirs(config_tuning.TUNING_TRIAL_MODELS_DIR, exist_ok=True)

    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(study_name=study_name, direction='maximize',
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
    TRAIN_END_DATE = '2015-01-01'
    # TEST_START_DATE = '2015-10-01'
    TEST_START_DATE = '2015-01-01'
    TEST_END_DATE = '2016-01-01'
    ticker_name_from_config_tickers = 'DOW_30_TICKER'
    # TRAIN_CSV_NAME = f'./data/train_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.csv'
    # TEST_CSV_NAME = f'./data/test_{ticker_name_from_config_tickers}_{TEST_START_DATE}_to_{TEST_END_DATE}.csv'

    # print('reading csvs...')
    # train = pd.read_csv(TRAIN_CSV_NAME, index_col='Unnamed: 0')
    # test = pd.read_csv(TEST_CSV_NAME, index_col='Unnamed: 0')
    # print(f'Train shape: {train.shape} \nTest shape: {test.shape}')
    processed_csv = './data/data_processed_DOW_30_TICKER_2009-01-01_to_2022-07-31.csv'
    print(f'Reading processed csv {processed_csv}')
    df_processed = pd.read_csv(processed_csv, index_col='Unnamed: 0')

    train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
    test = data_split(df_processed, TEST_START_DATE, TEST_END_DATE)
    
    # monthly
    if PERIOD == 'monthly':
        print('After converting to monthly...')
        train = sample_data_for_every_nth_day_of_the_month(df=train, date='02')
        test = sample_data_for_every_nth_day_of_the_month(df=test, date='02')
        print(f'Train shape: {train.shape} \nTest shape: {test.shape}')

    print('Train start date:', train['date'].iloc[0], ' Train end date:', train['date'].iloc[-1])
    print('Test start date:', test['date'].iloc[0], ' Test end date:', test['date'].iloc[-1])

    print(f'Train shape: {train.shape}, Test shape: {test.shape}')
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension
    env_kwargs = {
        "hmax": HMAX,
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

    tuned_hyperparams = get_tuned_hyperparams(train_df=train, test_df=test, env_kwargs=env_kwargs, study_name='daily_ddpg')
    print('Tuned hyperparams: ', tuned_hyperparams)