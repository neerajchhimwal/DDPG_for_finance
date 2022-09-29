from stock_trading_env import StockTradingEnv
import torch
import pandas as pd
import os
from config import RESULTS_DIR, BASELINE_TICKER_NAME_BACKTESTING

def get_prediction(model, environment, mode, initial_amount, cash_penalty=False):
    test_obs = environment.reset()
    account_memory = []
    actions_memory = []
    cumulative_reward_per_day = []
    print('env unique days: ', environment.df.index.nunique())
    
    # test_env.reset()
    for i in range(len(environment.df.index.unique())):
        with torch.no_grad():
            act = model.choose_action(test_obs, mode) # add mode='test'
        new_state, reward, done, info = environment.step(act)
        if cash_penalty:
            cumulative_reward = (environment.account_information["total_assets"][-1] - initial_amount) / initial_amount
        else:
            cumulative_reward = (environment.asset_memory[-1] - initial_amount) / initial_amount
        cumulative_reward_per_day.append(cumulative_reward)
        
        if i == (len(environment.df.index.unique()) - 1):
            account_memory = environment.save_asset_memory() 
            actions_memory = environment.save_action_memory()
            if cash_penalty:
                print('env cur step: ', environment.current_step)
                print('ac mem', account_memory)

        if done:
            if cash_penalty:
                print('i: ', i)
                print('env date index at done: ', environment.date_index)
            print("hit end!")
            break
    return account_memory, actions_memory, cumulative_reward_per_day

def trade_on_test_df(df, model, train_df, env_kwargs, mode='train', seed=0, trade_env=None, cash_penalty=False):
    insample_risk_indicator = train_df.drop_duplicates(subset=['date'])

    if BASELINE_TICKER_NAME_BACKTESTING == '^DJI':
        turb_threshold = insample_risk_indicator.turbulence.quantile(0.996)
    elif BASELINE_TICKER_NAME_BACKTESTING == 'NIFTY':
        turb_threshold = insample_risk_indicator.turbulence.quantile(0.984)
    elif BASELINE_TICKER_NAME_BACKTESTING == '^BSESN':
        turb_threshold = insample_risk_indicator.turbulence.quantile(0.97) #0.989
    turb_threshold = int(round(turb_threshold))
    
    env_kwargs = {k:v for (k,v) in env_kwargs.items() if k not in ["baseline_daily_returns"]}
    if not trade_env:
        trade_env = StockTradingEnv(df, turbulence_threshold=turb_threshold, risk_indicator_col='turbulence', **env_kwargs)
    trade_env.seed(seed)
    df_account_value, df_actions, last_day_cumulative_reward = get_prediction(model=model, 
                                                                            environment=trade_env, 
                                                                            initial_amount=env_kwargs['initial_amount'],
                                                                            mode=mode,
                                                                            cash_penalty=cash_penalty)

    return df_account_value, df_actions, last_day_cumulative_reward

def DRL_prediction_retraining(trade_data, model, iter_num, last_state, turbulence_threshold, initial, env_kwargs, seed):
    """make a prediction based on trained model"""

    ## trading env
    # trade_data = data_split(
    #     self.df,
    #     start=self.unique_trade_date[iter_num - self.rebalance_window],
    #     end=self.unique_trade_date[iter_num],
    # )
    trade_env = StockTradingEnv(
                                df=trade_data,
                                turbulence_threshold=turbulence_threshold,
                                risk_indicator_col='turbulence',
                                initial=initial,
                                previous_state=last_state,
                                **env_kwargs
                            )
    trade_env.seed(seed)
    trade_obs = trade_env.reset()

    account_memory = []
    actions_memory = []

    # test_env.reset()
    for i in range(len(trade_env.df.index.unique())):
        with torch.no_grad():
            act = model.choose_action(trade_obs)
        new_state, reward, done, info = trade_env.step(act)

        if i == (len(trade_env.df.index.unique()) - 2):
            account_memory = trade_env.save_asset_memory() 
            actions_memory = trade_env.save_action_memory()
            last_state = trade_env.render()

    df_last_state = pd.DataFrame({"last_state": last_state})
    df_last_state.to_csv(os.path.join(RESULTS_DIR, f"last_state_{iter_num}.csv"), index=False)
    return account_memory, actions_memory, last_state