from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
import wandb
from config import *

ENV_NAME = 'LunarLanderContinuous-v2'


w_config = dict(
  alpha = ACTOR_LR,
  beta = CRITIC_LR,
  batch_size = BATCH_SIZE,
  layer1_size = LAYER_1_SIZE,
  layer2_size = LAYER_2_SIZE,
  input_dims = STATE_SPACE,
  n_actions = N_ACTIONS,
  tau = TAU,
  architecture = "DDPG",
  env = ENV_NAME
)

PROJECT_NAME = f"pytorch_ddpg_{ENV_NAME.lower()}"

if USE_WANDB:
    run = wandb.init(project=PROJECT_NAME, tags=["DDPG", "RL"], config=w_config, job_type='train_model')


env = gym.make(ENV_NAME)
agent = Agent(alpha=ACTOR_LR, beta=CRITIC_LR, input_dims=STATE_SPACE, tau=TAU, env=ENV_NAME,
              batch_size=BATCH_SIZE, layer1_size=LAYER_1_SIZE, layer2_size=LAYER_2_SIZE, 
              n_actions=N_ACTIONS)

starting_episode = 0
if not TRAIN_FROM_SCRATCH:
    agent.load_models()
    starting_episode = agent.episode + 1

np.random.seed(0)

score_history = []
for i in range(starting_episode, TOTAL_EPISODES):
    obs = env.reset()
    done = False
    score = 0
    step_count = 0
    actor_loss_per_step_list = []
    critic_loss_per_step_list = []
    agent.episode = i
    while not done:
        act = agent.choose_action(obs)
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        step_count += 1 
        # run.log({'Actor Loss - STEP': agent.actor_loss_step})
        # run.log({'Critic Loss - STEP': agent.critic_loss_step}) 
        actor_loss_per_step_list.append(agent.actor_loss_step)
        critic_loss_per_step_list.append(agent.critic_loss_step)
        if step_count % 50 == 0:
            print(f'actor loss step {step_count}: {agent.actor_loss_step}')
            print(f'critic loss step {step_count}: {agent.critic_loss_step}')
        # this is a temporal diff method: we learn at each timestep, 
        # unline monte carlo methods where learning is done at the end of an episode
        score += reward
        obs = new_state
        # env.render()
    score_history.append(score)
    if USE_WANDB:
        run.log({'steps per episode': step_count, 'episode': i})
        run.log({'reward': score, 'episode': i})
        run.log({'reward avg 100 games': np.mean(score_history[-100:]), 'episode': i})
        run.log({'Actor loss': np.mean(actor_loss_per_step_list), 'episode': i})
        run.log({'Critic loss': np.mean(critic_loss_per_step_list), 'episode': i})

    if i % SAVE_CKP_AFTER_EVERY_NUM_EPISODES == 0:
       agent.save_models()
    print('='*50)
    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
    print('='*50)

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)