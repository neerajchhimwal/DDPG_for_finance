import matplotlib.pyplot as plt 
import numpy as np
import datetime
import pandas as pd
from plot import get_baseline_ind, get_comparison_df, backtest_plot, get_baseline, get_daily_return
from copy import deepcopy

def plotLearning(scores, filename, x=None, window=5):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')       
    plt.xlabel('Game')                     
    plt.plot(x, running_avg)
    plt.savefig(filename)

def trading_window_filtering(processed, window):
    start_date = processed.date.iloc[0]
    end_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d")+datetime.timedelta(days=window)).strftime('%Y-%m-%d')
    trading_days = []

    for i in range(len(processed.date)):
        if processed.date.iloc[i]>=start_date and processed.date.iloc[i]<=end_date:
            trading_days.append(end_date)
        elif  processed.date.iloc[i]>end_date:
            start_date=end_date
            end_date = (datetime.datetime.strptime(start_date, "%Y-%m-%d")+datetime.timedelta(days=window)).strftime('%Y-%m-%d')
            trading_days.append(end_date)

    processed['trading_timestemp'] = trading_days
    new_processed = processed[processed['date'].isin(processed['trading_timestemp'])]
    #new_processed = new_processed.drop(columns = ["index"])
    
    idx, i = [], 0
    cur_date = new_processed.date.iloc[0]
    for j in range(len(new_processed)):
        if new_processed.date.iloc[j] == cur_date:
            idx.append(i)
        else:
            cur_date = new_processed.date.iloc[j] 
            i += 1
            idx.append(i)

    new_processed = new_processed.set_axis(idx)

    return new_processed

def sample_data_for_every_nth_day_of_the_month(df, date):
    '''
    df: dataframe with column ['date'] in the format 'yyyy-mm-dd' [type: str]
    date: dd from 'yyyy-mm-dd' [type: str]
        eg: if date=='02':
                data for every 2nd of every month will be sampled. 2nd Jan, 2nd Feb etc.
    '''
    
    yyyy_mm_pairs = [dt[:-3] for dt in df['date']]
    dd_list = [dt[-2:] for dt in df['date']]
    df['yyyy_mm_pairs'] = yyyy_mm_pairs
    df['dd_list'] = dd_list

    indices_to_sample = []
    for ym in df['yyyy_mm_pairs'].unique():
        ym_df = df[df['yyyy_mm_pairs']==ym]
        dd_list_int = [int(i) for i in list(ym_df['dd_list'])]
    
        if int(date) in dd_list_int:
            indices_to_sample.append(ym_df[ym_df['dd_list']==date].index[0])
        else:
            new_date = [d for d in dd_list_int if d > int(date)][0]
            if len(str(new_date)) == 1:
                new_date = '0'+str(new_date)
            indices_to_sample.append(ym_df[ym_df['dd_list']==new_date].index[0])
            
    df = df.loc[indices_to_sample, :]
    
    idx, i = [], 0
    cur_date = df.date.iloc[0]
    for j in range(len(df)):
        if df.date.iloc[j] == cur_date:
            idx.append(i)
        else:
            cur_date = df.date.iloc[j] 
            i += 1
            idx.append(i)

    df = df.set_axis(idx)
    return df

def get_baseline_daily_returns(baseline_ticker, train):   
    if baseline_ticker=='^DJI':
        baseline_df = get_baseline(
                                    ticker=baseline_ticker, 
                                    start=train.date.iloc[0],
                                    end=train.date.iloc[-1]
                                )
    else:
        baseline_df = get_baseline_ind(
                                    ticker=baseline_ticker, 
                                    start=train.date.iloc[0],
                                    end=train.date.iloc[-1]
        )

    train_cpy = deepcopy(train)
    train_cpy = train_cpy.drop_duplicates(subset=['date'])
    train_cpy["date"] = pd.to_datetime(train_cpy["date"])
    baseline_df["date"] = pd.to_datetime(baseline_df["date"], format="%Y-%m-%d")
    baseline_df = pd.merge(train_cpy[["date"]], baseline_df, how="left", on="date")
    baseline_df = baseline_df.fillna(method="ffill").fillna(method="bfill")
    baseline_returns = get_daily_return(baseline_df, value_col_name="close")
    baseline_daily_returns_wrt_day = pd.DataFrame(baseline_returns)
    # df containing % return compared to previous day's portfolio amt
    baseline_daily_returns_wrt_day['day'] = range(0, len(baseline_daily_returns_wrt_day)) 
    baseline_daily_returns_wrt_day.fillna(value=0, inplace=True)
    baseline_daily_returns_wrt_day_list = list(baseline_daily_returns_wrt_day['daily_return'])
    return baseline_daily_returns_wrt_day_list

def get_df_cum_ret(cum_rets):
    d = pd.DataFrame(cum_rets)
    d = d.reset_index('date')
    d['date'] = [i.date() for i in d['date']]
    d['date'] = [str(i) for i in d['date']]
    d.set_index('date', inplace=True)
    d.rename(columns = {0:'Cumulative Returns'}, inplace = True)
    d['Cumulative Returns'] = [i*100 for i in d['Cumulative Returns']]
    d.rename(columns={'Cumulative Returns': 'Cumulative Returns %'}, inplace=True)
    return d