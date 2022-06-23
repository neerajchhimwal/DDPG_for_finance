import torch

ACTOR_LR = 0.000025
CRITIC_LR = 0.00025
BATCH_SIZE = 64
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300
TAU = 0.001
N_ACTIONS = 2
STATE_SPACE = [8]

TRAIN_FROM_SCRATCH = True
CHECKPOINT_DIR = 'tmp/ddpg/'

TOTAL_EPISODES = 2500
SAVE_CKP_AFTER_EVERY_NUM_EPISODES = 25

USE_WANDB = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Ornstein Uhlenbeck noise parameters
sigma = 0.15 
theta = 0.2 
dt = 1e-2