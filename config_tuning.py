from config import PERIOD

if PERIOD == 'monthly':
    N_TRIALS = 15  # number of HP optimization runs
    TOTAL_EPISODES = 20 # per HP optimization run
else:
    N_TRIALS = 15  # number of HP optimization runs
    TOTAL_EPISODES = 5 # per HP optimization run

SAVE_MODELS = False
TUNING_TRIAL_MODELS_DIR = './models_t' # only applicable when SAVE_MODELS=True