import config, config_tickers
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

import numpy as np
import pandas as pd
import itertools

# def download(start_date, end_date, ticker_list=config_tickers.DOW_30_TICKER):
#     df = YahooDownloader(start_date = start_date, end_date = start_date, ticker_list = ticker_list).fetch_data()

#     return df


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