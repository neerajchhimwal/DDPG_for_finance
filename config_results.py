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
ACTOR_LR = 0.008767
CRITIC_LR = 0.008767
BATCH_SIZE = 128
LAYER_1_SIZE = 512
LAYER_2_SIZE = 512
# LAYER_3_SIZE = 512
TAU = 0.001
BUFFER_SIZE = 100000
SEED = 12321
# SEED = 9927

# n_actions and state_space dims depend on data and are being overwritten in training file
N_ACTIONS = 2
STATE_SPACE = 8

# PERIOD = 'monthly'
PERIOD = 'daily'
DATE_OF_THE_MONTH_TO_TAKE_ACTIONS = '02' # this will be used only when PERIOD == 'monthly'

CHECKPOINT_DIR = f'./trained_models/run'
SAVED_CHECKPOINT_PATH = './trained_models/run/agent_ep_49.pt'
RESULTS_DIR = f'./results/test'

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2016-01-01'

TEST_START_DATE = '2016-01-01'
TEST_END_DATE = '2020-05-09'

TRADE_START_DATE = TEST_START_DATE
TRADE_END_DATE = '2022-01-04'

TRADE_2_START_DATE = TEST_START_DATE
TRADE_2_END_DATE = '2022-01-04'


BASELINE_TICKER_NAME_BACKTESTING = '^DJI'
# BASELINE_TICKER_NAME_BACKTESTING = '^BSESN'

INITIAL_AMOUNT = 1000000 # USD

if PERIOD == 'monthly':
    HMAX = 500
elif PERIOD == 'daily':
    HMAX = 100

USE_WANDB = False
PROJECT_NAME = 'dji_daily' # wandb project name

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
            'close_60_sma',
            # 'log-ret', 
            # 'chop',
            # 'ppo',
            # 'stochrsi',
            # 'close_7_smma',
            # 'vr',
            # 'wr'
]

UNIFORM_INITIAL_EXPLORATION = False 
EXPLORATION_STEP_COUNT = 0 # 20000 # keep this 0 when not using UNIFORM INITIAL EXPLORATION
NOISE_ANNEALING = False
EXPLORATION_NOISE = "OUACTION"
# EXPLORATION_NOISE = "GAUSSIAN"
# Ornstein Uhlenbeck noise parameters
sigma = 0.15 
theta = 0.2
dt = 1e-2

NORMAL_SCALAR = 0.1 # std dev if using gaussian noise

# lr
LR_SCHEDULE_STEP_SIZE_ACTOR = 20 
LR_SCHEDULE_STEP_SIZE_CRITIC = 20

