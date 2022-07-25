import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json

from config import DEVICE, sigma, theta, dt
from config import CHECKPOINT_DIR, LR_SCHEDULE_STEP_SIZE
import os
import wandb

class OUActionNoise(object):
    '''
    implements noise to model the physics of a running particle,
        "random normal noise correlated in time"

    this will be added in the actor class to add some exploration noise
    '''
    def __init__(self, mu, sigma=sigma, theta=theta, dt=dt, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset() # resets the temporal correlation

    def __call__(self):
        '''
        Dunder mothod that allows you to create object like this:
                            noise = OUActionNoise()
                            noise() 
        instead of noise.get_noise() or something similar
        '''
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        '''
        returns printable representation of an object.
        Usage: repr(object)
        '''
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class ReplayBuffer(object):
    '''
    memory to keep track of (state, action, reward) transitions, so it can be sampled during learning
    '''
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

class CriticNetwork(nn.Module):
    '''
    approximates the value function for critic network
    '''
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir=CHECKPOINT_DIR):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_lr_'+str(beta)+'_ddpg.pt')
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        # below 3 lines to initilaize the starting weights of parameters in a very narrow region space: helps with comvergence
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims) # batch normalization helps with convergence

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True, factor=0.1)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_SCHEDULE_STEP_SIZE, gamma=0.5)

        self.device = T.device(DEVICE)
        self.episode = 0
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)  
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value)) # order matters when adding in relu. can take separate relus and then add?
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, episode):
        print('... saving checkpoint ...')

        state = {
                'state': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'episode': episode
        }

        T.save(state, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode = checkpoint['episode']
        self.to(T.device(DEVICE))
        

class ActorNetwork(nn.Module):
    '''
    approximates the policy function
    '''
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir=CHECKPOINT_DIR):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_lr_'+str(alpha)+'_ddpg.pt')
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.n_actions) # mu represents actions here
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_SCHEDULE_STEP_SIZE, gamma=0.5)

        self.device = T.device(DEVICE)
        self.episode = 0
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = T.tanh(self.mu(x)) # hyperboplic tangent binds the action in the range -1 to +1 (from the paper)
                               # for use cases where actions need to be in a different range, multiplication is required

        return x

    def save_checkpoint(self, episode):
        print('... saving checkpoint ...')

        state = {
                'state': self.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'episode': episode
        }

        T.save(state, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        checkpoint = T.load(self.checkpoint_file)
        self.load_state_dict(checkpoint['state'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.episode = checkpoint['episode']
        self.to(T.device(DEVICE))
        

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, ckp_dir, gamma=0.99,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=64):

        '''
        alpha and beta are the learning rates of actor and critic n/w respectively
        '''
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.checkpoint_dir = ckp_dir
        self.actor = ActorNetwork(alpha, input_dims, layer1_size,
                                  layer2_size, n_actions=n_actions,
                                  name='Actor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size,
                                    layer2_size, n_actions=n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size,
                                         layer2_size, n_actions=n_actions,
                                         name='TargetActor')

        self.target_critic = CriticNetwork(beta, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions,
                                           name='TargetCritic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.update_network_parameters(tau=1)
        self.actor_loss_step = 0
        self.critic_loss_step = 0
        self.episode = 0

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation)
        self.actor.train()
        exp_noise = T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        mu += exp_noise
        mu = T.clamp(mu, min=-1, max=1)
        # with T.no_grad():
        #     mu = self.actor.forward(observation).cpu().detach().numpy()
        # exp_noise = self.noise()

        # mu_prime = np.clip(mu + exp_noise, -1, +1)
        
        # mu_prime = T.clip(mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device), min=-1, max=+1)
        # mu_prime = T.tanh(mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device))
        # mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        # print(f'mu: {mu} mu_prime: {mu_prime}') 
        # mu_prime.to(self.actor.device)
        
        return mu.cpu().detach().numpy()


    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)
                                      
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        # with T.no_grad():
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()

        critic_loss = F.mse_loss(critic_value, target)
        
        critic_loss_copy = critic_loss.detach().clone()
        self.critic_loss_step = critic_loss_copy.cpu().numpy()

        # backward
        # self.critic.optimizer.zero_grad()
        critic_loss.backward()

        # grad clip
        clipping_value = 1
        T.nn.utils.clip_grad_norm_(self.critic.parameters(), clipping_value)
        # step
        self.critic.optimizer.step()
        self.critic.eval()
        
        mu = self.actor.forward(state)
        self.actor.train()
        # actor_loss = -self.critic.forward(state, mu)
        # actor_loss = T.mean(actor_loss)
        actor_gradients = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_gradients)

        # backward
        # self.actor.optimizer.zero_grad()
        actor_loss.backward()

        # grad clip
        T.nn.utils.clip_grad_norm_(self.actor.parameters(), clipping_value)

        # step                                   
        self.actor.optimizer.step()

        actor_loss_copy = actor_loss.detach().clone()
        self.actor_loss_step = actor_loss_copy.cpu().numpy()

        self.update_network_parameters()
        

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        target_critic_dict_temp = target_critic_dict.copy()
        target_actor_dict_temp = target_actor_dict.copy()
    
        for name in critic_state_dict:
            target_critic_dict_temp[name] = tau*critic_state_dict[name] + \
                                            (1-tau)*target_critic_dict[name]

        self.target_critic.load_state_dict(target_critic_dict_temp)

        for name in actor_state_dict:
            target_actor_dict_temp[name] = tau*actor_state_dict[name] + \
                                      (1-tau)*target_actor_dict[name]

        self.target_actor.load_state_dict(target_actor_dict_temp)


    def save_models(self):
        '''
        Saving state dicts as well as optimizer state for each model
        '''

        episode = self.episode
        
        self.actor.save_checkpoint(episode)
        self.target_actor.save_checkpoint(episode)
        self.critic.save_checkpoint(episode)
        self.target_critic.save_checkpoint(episode)

    def load_models(self):
        '''
        also load episode number here
        '''
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
        self.episode = self.target_critic.episode

    def save_checkpoint(self, last_episode, checkpoint_name=None, save_noise=False):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        
        """
        if checkpoint_name is None:
            checkpoint_name = os.path.join(self.checkpoint_dir, 'agent_ep_{}.pt'.format(last_episode))
        else:
            checkpoint_name = os.path.join(self.checkpoint_dir, checkpoint_name)
        if save_noise:
            noise_name = os.path.join(self.checkpoint_dir, 'noise_ep_{}.npy'.format(last_episode))

        print('Saving checkpoint...')
        checkpoint = {
            'last_episode': last_episode,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
            'replay_buffer': self.memory
        }
        print('Saving agent at episode {}...'.format(last_episode))
        T.save(checkpoint, checkpoint_name)

        if save_noise:
            print('Saving noise at episode {} as {}...'.format(last_episode, noise_name))
            np.save(noise_name, self.noise)

    def get_paths_of_latest_files(self, load_noise=False):
        """
        Returns the latest created agent and noise files in 'checkpoint_dir'
        """
        ckp_files = [file for file in os.listdir(self.checkpoint_dir) if file.endswith(".pt")]
        ckp_filepaths = [os.path.join(self.checkpoint_dir, file) for file in ckp_files]
        last_ckp_file = max(ckp_filepaths, key=os.path.getctime)

        noise_p = ''
        if load_noise:
            noise_files = [file for file in os.listdir(self.checkpoint_dir) if file.endswith(".npy")]
            noise_filepaths = [os.path.join(self.checkpoint_dir, file) for file in noise_files]
            last_noise_file = max(noise_filepaths, key=os.path.getctime)
            noise_p = os.path.abspath(last_noise_file)

        return os.path.abspath(last_ckp_file), noise_p

    def load_checkpoint(self, checkpoint_path=None, load_noise=False):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.
        Arguments:
            checkpoint_path:    File to load the model from
        """

        if checkpoint_path is None:
            checkpoint_path, noise_path = self.get_paths_of_latest_files(load_noise)

        if os.path.isfile(checkpoint_path):
            print("Loading checkpoint...({})".format(checkpoint_path))

            checkpoint = T.load(checkpoint_path, map_location=DEVICE)
            start_episode = checkpoint['last_episode'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.target_actor.load_state_dict(checkpoint['target_actor'])
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.memory = checkpoint['replay_buffer']

            if load_noise:
                self.noise = np.load(noise_path)
                print('Loaded noise from {}'.format(noise_path))

            print('Loaded model at episode {} from {}'.format(start_episode, checkpoint_path))
            return start_episode
        else:
            raise OSError('Checkpoint not found')

    def set_train(self):
        self.actor.train()
        self.critic.train()
        self.target_actor.train()
        self.target_critic.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.target_actor.eval()
        self.target_critic.eval()

    def train_model(self, 
                    total_episodes, 
                    train_from_scratch, 
                    env, 
                    env_kwargs, 
                    ckp_save_freq, 
                    predict_on_test=False, 
                    test_reward_table_save_freq=10,
                    use_wandb=False,
                    wandb_config=None,
                    wandb_project_name=None,
                    save_ckp=False):

        if use_wandb:
            run = wandb.init(project=wandb_project_name, tags=["DDPG", "RL"], config=wandb_config, job_type='train_model')

        starting_episode = 0
        if not train_from_scratch:
            starting_episode = self.load_checkpoint()
        
        day_counter_total = 0
        score_history = []
        for i in range(starting_episode, total_episodes):
            obs = env.reset()
            self.noise.reset()
            done = False
            score = 0
            step_count = 0
            actor_loss_per_episode = 0
            critic_loss_per_episode = 0
            # actor_loss_per_episode = []
            # critic_loss_per_episode = []
            self.episode = i
            cumulative_rewards_per_step_this_episode = []
            while not done:
                act = self.choose_action(obs)
                new_state, reward, done, info = env.step(act)
                self.remember(obs, act, reward, new_state, int(done))
                self.learn()
                step_count += 1
                cumulative_reward = (env.asset_memory[-1] - env_kwargs['initial_amount']) / env_kwargs['initial_amount']
                cumulative_rewards_per_step_this_episode.append(cumulative_reward)
                if use_wandb:
                    run.log({'Cumulative returns': cumulative_reward, 'days':day_counter_total})
                day_counter_total += 1
                
                actor_loss_per_episode += self.actor_loss_step
                critic_loss_per_episode += self.critic_loss_step
                score += reward
                obs = new_state

            # print('$'*100)
            # cr_lr = self.critic.optimizer.state_dict()['param_groups'][0]['lr']
            # ac_lr = self.actor.optimizer.state_dict()['param_groups'][0]['lr']
            # print(f"Episode {i}, critic LR: {cr_lr}, critic loss: {critic_loss_per_episode}")
            # print(f"Episode {i},  actor LR: {ac_lr},   actor loss: {actor_loss_per_episode}")
            # print('$'*100)
            
            self.critic.scheduler.step()
            self.actor.scheduler.step()

            score_history.append(score)
            if use_wandb:
                run.log({'steps per episode': step_count, 'episode': i})
                run.log({'reward': score, 'episode': i})
                # run.log({'reward avg 100 games': np.mean(score_history[-100:]), 'episode': i})
                run.log({'Actor loss': actor_loss_per_episode, 'episode': i})
                run.log({'Critic loss': critic_loss_per_episode, 'episode': i})

                # run.log({'Actor LR': ac_lr, 'episode': i})
                # run.log({'Critic LR': cr_lr, 'episode': i})
            
            if save_ckp:
                if (i % ckp_save_freq == 0 or i == total_episodes-1):
                    self.save_checkpoint(last_episode=i)

        return self
