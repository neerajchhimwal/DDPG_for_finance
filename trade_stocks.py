from stock_trading_env import StockTradingEnv


def get_prediction(model, environment, initial_amount):
    test_obs = environment.reset()
    account_memory = []
    actions_memory = []
    cumulative_reward_per_day = []

    # test_env.reset()
    for i in range(len(environment.df.index.unique())):
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

def trade_on_test_df(df, model, train_df, env_kwargs):
    insample_risk_indicator = train_df.drop_duplicates(subset=['date'])
    turb_threshold = insample_risk_indicator.turbulence.quantile(0.996)
    turb_threshold = int(round(turb_threshold))
    
    trade_env = StockTradingEnv(df, turbulence_threshold=turb_threshold, risk_indicator_col='turbulence', **env_kwargs)
    df_account_value, df_actions, last_day_cumulative_reward = get_prediction(model=model, 
                                                                            environment=trade_env, 
                                                                            initial_amount=env_kwargs['initial_amount'])

    return df_account_value, df_actions, last_day_cumulative_reward
