import config, config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

from nsepy import get_history
from datetime import date

# def download(start_date, end_date, ticker_list=config_tickers.DOW_30_TICKER):
#     df = YahooDownloader(start_date = start_date, end_date = start_date, ticker_list = ticker_list).fetch_data()

#     return df

def get_date_from_str(str_date):
    '''
    str_date format: 'yyyy-mm-dd'
    '''
    return date(int(str_date.split('-')[0]), int(str_date.split('-')[1]), int(str_date.split('-')[2]))

def normalise_tic_name_to_latest(df):
    if df['Symbol'].nunique() != 1:
        latest_tic = df.iloc[-1]['Symbol']
        
        # setting latest tic across all rows
        df['Symbol'] = [latest_tic]*len(df)
    return df

def download_indian_stocks(ticker_list, start_date, end_date):
    '''
    start_date: str in format 'yyyy-mm-dd'
    '''
    dfs = []
    for sym in tqdm(ticker_list):
        df = get_history(symbol=sym, start=get_date_from_str(start_date), end=get_date_from_str(end_date))
        # print(sym)
        if df['Symbol'].nunique() != 1:
            df = normalise_tic_name_to_latest(df)
        dfs.append(df)
    final_df = pd.concat(dfs)

    final_df.rename(columns = {'Symbol':'tic', 'Close':'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume':'volume'}, inplace = True)
    final_df = final_df.rename_axis('date').reset_index()
    final_df = final_df.sort_values(['date', 'tic'])
    final_df.drop_duplicates(inplace=True)
    final_df.reset_index(drop=True, inplace=True)

    return final_df

def process_data(df, use_technical_indicator=True, technical_indicator_list=config.INDICATORS, use_vix=True, use_turbulence=True, user_defined_feature=False):

    fe = FeatureEngineer(
                            use_technical_indicator=use_technical_indicator,
                            tech_indicator_list=technical_indicator_list,
                            use_vix=use_vix,
                            use_turbulence=use_turbulence,
                            user_defined_feature=user_defined_feature
                        )
    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    return processed_full