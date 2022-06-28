import torch
import os

ACTOR_LR = 0.0025
CRITIC_LR = 0.0025
BATCH_SIZE = 128
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300
TAU = 0.001


# n_actions and state_space dims depend on data and are being overwritten in training file
N_ACTIONS = 2
STATE_SPACE = 8

TRAIN_FROM_SCRATCH = True
SEED = 0
CHECKPOINT_DIR = 'trained_models/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2016-01-01'
VALID_START_DATE = '2016-01-02'
VALID_END_DATE = '2022-07-31'

ticker_name_from_config_tickers = 'DOW_30_TICKER'
ORIGINAL_CSV_NAME = f'./data/data_raw_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{VALID_END_DATE}.csv'
PROCESSED_CSV_NAME = f'./data/data_processed_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{VALID_END_DATE}.csv'
TRAIN_CSV_NAME = f'./data/train_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.csv'
TRADE_CSV_NAME = f'./data/trade_{ticker_name_from_config_tickers}_{VALID_START_DATE}_to_{VALID_END_DATE}.csv'

TOTAL_EPISODES = 100
SAVE_CKP_AFTER_EVERY_NUM_EPISODES = 10

USE_WANDB = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Finance config
INDICATORS = [
            'macd',
            'boll_ub',
            'boll_lb',
            'rsi_30',
            'cci_30',
            'dx_30',
            'close_30_sma',
            'close_60_sma'
]


# Ornstein Uhlenbeck noise parameters
sigma = 0.15 
theta = 0.2 
dt = 1e-2