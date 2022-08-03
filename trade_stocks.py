from stock_trading_env import StockTradingEnv
import torch
import pandas as pd
import os
from config import RESULTS_DIR

def get_prediction(model, environment, initial_amount):
    test_obs = environment.reset()
    account_memory = []
    actions_memory = []
    cumulative_reward_per_day = []

    # test_env.reset()
    for i in range(len(environment.df.index.unique())):
        with torch.no_grad():
            act = model.choose_action(test_obs)
        new_state, reward, done, info = environment.step(act)
        cumulative_reward = (environment.asset_memory[-1] - initial_amount) / initial_amount
        cumulative_reward_per_day.append(cumulative_reward)
        if i == (len(environment.df.index.unique()) - 2):
            account_memory = environment.save_asset_memory() 
            actions_memory = environment.save_action_memory()

        if done:
            print("hit end!")
            break
    return account_memory, actions_memory, cumulative_reward_per_day

def trade_on_test_df(df, model, train_df, env_kwargs, seed=0):
    insample_risk_indicator = train_df.drop_duplicates(subset=['date'])
    turb_threshold = insample_risk_indicator.turbulence.quantile(0.996)
    turb_threshold = int(round(turb_threshold))
    
    trade_env = StockTradingEnv(df, turbulence_threshold=turb_threshold, risk_indicator_col='turbulence', **env_kwargs)
    trade_env.seed(seed)
    df_account_value, df_actions, last_day_cumulative_reward = get_prediction(model=model, 
                                                                            environment=trade_env, 
                                                                            initial_amount=env_kwargs['initial_amount'])

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