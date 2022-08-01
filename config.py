import torch
import os

# ACTOR_LR = 0.1946789720401594
# CRITIC_LR = 0.1946789720401594
ACTOR_LR = 0.0031060995478059458
CRITIC_LR = 0.0031060995478059458
BATCH_SIZE = 8
LAYER_1_SIZE = 512
LAYER_2_SIZE = 512
TAU = 0.001
BUFFER_SIZE = 100000

# n_actions and state_space dims depend on data and are being overwritten in training file
N_ACTIONS = 2
STATE_SPACE = 8

TRAIN_FROM_SCRATCH = True
# SEED = 12321
# SEED = 329
# SEED = 333
SEED = 9923
# SEED = 0
# SEED = 42

CHECKPOINT_DIR = f'trained_models/monthly_data_seed_{SEED}_run_2/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2016-01-01'
# TRAIN_END_DATE = '2020-06-30'

TEST_START_DATE = '2016-01-01'
TEST_END_DATE = '2020-05-10'

#  2020/07/01 to 2021/06/30
TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2022-07-31'

ticker_name_from_config_tickers = 'DOW_30_TICKER'
ORIGINAL_CSV_NAME = f'./data/data_raw_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRADE_END_DATE}.csv'
PROCESSED_CSV_NAME = f'./data/data_processed_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRADE_END_DATE}.csv'
TRAIN_CSV_NAME = f'./data/train_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.csv'
TEST_CSV_NAME =  f'./data/test_{ticker_name_from_config_tickers}_{TEST_START_DATE}_to_{TEST_END_DATE}.csv'
TRADE_CSV_NAME = f'./data/trade_{ticker_name_from_config_tickers}_{TRADE_START_DATE}_to_{TRADE_END_DATE}.csv'

USE_MONTHLY_DATA = True
DATE_OF_THE_MONTH_TO_TAKE_ACTIONS = '02' # this will be used only when USE_DAILY_DATA == False

BASELINE_TICKER_NAME_BACKTESTING = '^DJI'

TOTAL_EPISODES = 200
SAVE_CKP_AFTER_EVERY_NUM_EPISODES = 150
SAVE_REWARD_TABLE_AFTER_EVERY_NUM_EPISODES = 10

USE_WANDB = True
# RESUME_LAST_WANDB_RUN = True

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

# lr
LR_SCHEDULE_STEP_SIZE = 10