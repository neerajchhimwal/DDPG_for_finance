from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
import wandb

ENV_NAME = 'LunarLanderContinuous-v2'
ACTOR_LR = 0.000025
CRITIC_LR = 0.00025
BATCH_SIZE = 64
LAYER_1_SIZE = 400
LAYER_2_SIZE = 300

config = dict(
  alpha = ACTOR_LR,
  beta = CRITIC_LR,
  batch_size = BATCH_SIZE,
  layer1_size = LAYER_1_SIZE,
  layer2_size = LAYER_2_SIZE,
  input_dims = [8],
  n_actions = 2,
  tau = 0.001,
  architecture = "DDPG",
  infra = "Ubuntu",
  env = ENV_NAME
)
PROJECT_NAME = f"pytorch_ddpg_{ENV_NAME.lower()}"

run = wandb.init(project=PROJECT_NAME, tags=["DDPG", "FCL", "RL"], config=config, job_type='train_model')


env = gym.make(ENV_NAME)
agent = Agent(alpha=config['alpha'], beta=config['beta'], input_dims=config['input_dims'], tau=config['tau'], env=config['env'],
              batch_size=config['batch_size'], layer1_size=config['layer1_size'], layer2_size=config['layer2_size'], 
              n_actions=config['n_actions'])

# uncomment the load_models line if you're testing and want to continue using trained models
# agent.load_models()
np.random.seed(0)

score_history = []
for i in range(2000): # 1000 episodes (here, games)
    obs = env.reset()
    done = False
    score = 0
    step_count = 0
    actor_loss_per_step_list = []
    critic_loss_per_step_list = []
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

    run.log({'steps per episode': step_count, 'episode': i})
    run.log({'reward': score, 'episode': i})
    run.log({'reward avg 100 games': np.mean(score_history[-100:]), 'episode': i})
    run.log({'Actor loss': np.mean(actor_loss_per_step_list), 'episode': i})
    run.log({'Critic loss': np.mean(critic_loss_per_step_list), 'episode': i})

    if i % 25 == 0:
       agent.save_models()
    print('='*50)
    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
    print('='*50)

filename = 'LunarLander-alpha000025-beta00025-400-300.png'
plotLearning(score_history, filename, window=100)