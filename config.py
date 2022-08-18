import torch
import os

## monthly params
# ACTOR_LR = 0.0011996013093168115
# CRITIC_LR = 0.0011996013093168115
# BATCH_SIZE = 8
# LAYER_1_SIZE = 512
# LAYER_2_SIZE = 512
# TAU = 0.001
# BUFFER_SIZE = 100000
# SEED = 76282

# daily params
ACTOR_LR = 0.00395461772605366
CRITIC_LR = 0.00395461772605366
BATCH_SIZE = 256
LAYER_1_SIZE = 512
LAYER_2_SIZE = 512
TAU = 0.001
BUFFER_SIZE = 100000
# SEED = 42
SEED = 12321


# n_actions and state_space dims depend on data and are being overwritten in training file
N_ACTIONS = 2
STATE_SPACE = 8

# PERIOD = 'monthly'
PERIOD = 'daily'
DATE_OF_THE_MONTH_TO_TAKE_ACTIONS = '02' # this will be used only when PERIOD == 'monthly'


TRAIN_FROM_SCRATCH = True
RETRAIN_IN_MONTHS = 12  # only applicable when retraining after every n months 
TEST_PERIOD_IN_MONTHS = 12 # only applicable when doing iterative hyperparam tuning while retraining

CHECKPOINT_DIR = f'./trained_models/dji_daily'
RESULTS_DIR = f'./results/dji_daily'


if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2016-01-01'

TEST_START_DATE = '2016-01-01'
TEST_END_DATE = '2020-05-10'

TRADE_START_DATE = '2016-01-01'
TRADE_END_DATE = '2022-01-04'

TRADE_2_START_DATE = '2016-01-01'
TRADE_2_END_DATE = '2022-07-31'

# these CSV names are not being used
ticker_name_from_config_tickers = 'DOW_30_TICKER'
ORIGINAL_CSV_NAME = f'./data/data_raw_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRADE_END_DATE}.csv'
PROCESSED_CSV_NAME = f'./data/data_processed_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRADE_END_DATE}.csv'
TRAIN_CSV_NAME = f'./data/train_{ticker_name_from_config_tickers}_{TRAIN_START_DATE}_to_{TRAIN_END_DATE}.csv'
TEST_CSV_NAME =  f'./data/test_{ticker_name_from_config_tickers}_{TEST_START_DATE}_to_{TEST_END_DATE}.csv'
TRADE_CSV_NAME = f'./data/trade_{ticker_name_from_config_tickers}_{TRADE_START_DATE}_to_{TRADE_END_DATE}.csv'


BASELINE_TICKER_NAME_BACKTESTING = '^DJI'
# BASELINE_TICKER_NAME_BACKTESTING = 'NIFTY'

if BASELINE_TICKER_NAME_BACKTESTING == 'NIFTY':
    INITIAL_AMOUNT = 100000000 # INR
else:
    INITIAL_AMOUNT = 1000000 # USD

if PERIOD == 'monthly':
    TOTAL_EPISODES = 250
else:
    TOTAL_EPISODES = 50

if PERIOD == 'monthly':
    HMAX = 500
elif PERIOD == 'daily':
    HMAX = 100



SAVE_CKP_AFTER_EVERY_NUM_EPISODES = TOTAL_EPISODES
# SAVE_REWARD_TABLE_AFTER_EVERY_NUM_EPISODES = 10

USE_WANDB = True

PROJECT_NAME = 'dji_daily'
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