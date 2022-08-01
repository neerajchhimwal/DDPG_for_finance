from copy import deepcopy

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyfolio
from pyfolio import timeseries
import empyrical as ep
# from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


def get_daily_return(df, value_col_name="account_value"):
    df = deepcopy(df)
    df["daily_return"] = df[value_col_name].pct_change(1)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True, drop=True)
    df.index = df.index.tz_localize("UTC")
    return pd.Series(df["daily_return"], index=df.index)


def convert_daily_return_to_pyfolio_ts(df):
    strategy_ret = df.copy()
    strategy_ret["date"] = pd.to_datetime(strategy_ret["date"])
    strategy_ret.set_index("date", drop=False, inplace=True)
    strategy_ret.index = strategy_ret.index.tz_localize("UTC")
    del strategy_ret["date"]
    return pd.Series(strategy_ret["daily_return"].values, index=strategy_ret.index)


def backtest_stats(account_value, value_col_name="account_value", period="daily"):
    dr_test = get_daily_return(account_value, value_col_name=value_col_name)
    
    if period == 'daily':
        perf_stats_all = timeseries.perf_stats(
            returns=dr_test,
            positions=None,
            transactions=None,
            turnover_denom="AGB",
        )
        print(perf_stats_all)
        
    else:
        returns = dr_test
        STATS_DICT = {
            'Annual return': ep.annual_return(returns, period=period),
            'Cumulative returns': ep.cum_returns_final(returns, starting_value=0),
            'Annual volatility': ep.annual_volatility(returns, period=period),
            'Sharpe ratio': ep.sharpe_ratio(returns, risk_free=0, period=period),
            'Calmar ratio': ep.calmar_ratio(returns, period=period),
            'Stability': ep.stability_of_timeseries(returns),
            'Max drawdown': ep.max_drawdown(returns),
            'Omega ratio': ep.omega_ratio(returns, required_return=0.0),
            'Sortino ratio': ep.sortino_ratio(returns, required_return=0, period=period),
            'Skew': None,
            'Kurtosis': None,
            'Tail ratio': ep.tail_ratio(returns),
            f'{period} value at risk': timeseries.value_at_risk(returns, period=period),

        }

        perf_stats_all = pd.Series(STATS_DICT)
        print(perf_stats_all)
        
    return perf_stats_all


def backtest_plot(
        account_value,
        baseline_start,
        baseline_end,
        baseline_ticker="^DJI",
        value_col_name="account_value",
):
    df = deepcopy(account_value)
    df["date"] = pd.to_datetime(df["date"])
    test_returns = get_daily_return(df, value_col_name=value_col_name)

    baseline_df = get_baseline(
        ticker=baseline_ticker, start=baseline_start, end=baseline_end
    )

    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(df[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")

    # with pyfolio.plotting.plotting_context(font_scale=1.1):
    #     pyfolio.create_full_tear_sheet(
    #         returns=test_returns, benchmark_rets=baseline_returns, set_context=False
    #     )

    return test_returns, baseline_returns

def get_baseline(ticker, start, end):
    return YahooDownloader(
        start_date=start, end_date=end, ticker_list=[ticker]
    ).fetch_data()


def trx_plot(df_trade, df_actions, ticker_list):
    df_trx = pd.DataFrame(np.array(df_actions["transactions"].to_list()))
    df_trx.columns = ticker_list
    df_trx.index = df_actions["date"]
    df_trx.index.name = ""

    for i in range(df_trx.shape[1]):
        df_trx_temp = df_trx.iloc[:, i]
        df_trx_temp_sign = np.sign(df_trx_temp)
        buying_signal = df_trx_temp_sign.apply(lambda x: x > 0)
        selling_signal = df_trx_temp_sign.apply(lambda x: x < 0)

        tic_plot = df_trade[
            (df_trade["tic"] == df_trx_temp.name)
            & (df_trade["date"].isin(df_trx.index))
            ]["close"]
        tic_plot.index = df_trx_temp.index

        plt.figure(figsize=(10, 8))
        plt.plot(tic_plot, color="g", lw=2.0)
        plt.plot(
            tic_plot,
            "^",
            markersize=10,
            color="m",
            label="buying signal",
            markevery=buying_signal,
        )
        plt.plot(
            tic_plot,
            "v",
            markersize=10,
            color="k",
            label="selling signal",
            markevery=selling_signal,
        )
        plt.title(
            f"{df_trx_temp.name} Num Transactions: {len(buying_signal[buying_signal == True]) + len(selling_signal[selling_signal == True])}"
        )
        plt.legend()
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
        plt.xticks(rotation=45, ha="right")
        plt.show()

def get_comparison_df(df_account_value, baseline_ticker, period="daily"):
    dates = str(df_account_value.loc[0,'date']) + '_to_' + str(df_account_value.loc[len(df_account_value)-1,'date'])
    agent_col = f'agent_{dates}'
    agent_results = pd.DataFrame(backtest_stats(account_value=df_account_value, period=period), columns=[agent_col])
    agent_results['metric'] = agent_results.index
    agent_results.reset_index(drop=True, inplace=True)

    baseline_df_dji = get_baseline(
        ticker=baseline_ticker, 
        start = df_account_value.loc[0,'date'],
        end = df_account_value.loc[len(df_account_value)-1,'date'])

    baseline_indices = [i for i, date in enumerate(baseline_df_dji['date']) if date in list(df_account_value['date'])]
    new_baseline_df = baseline_df_dji.loc[baseline_indices, :]
    dji_col = f'dji_{dates}'
    dji_results = pd.DataFrame(backtest_stats(account_value=new_baseline_df, value_col_name="close", period=period), columns=[dji_col])
    dji_results['metric'] = dji_results.index
    dji_results.reset_index(drop=True, inplace=True)
    results_df = agent_results.merge(dji_results, on='metric')
    results_df = results_df[['metric', agent_col, dji_col]]

    return results_df