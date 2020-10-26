import numpy as np
import torch
from scipy.special import expit, logit           # sigmoid function
import wandb

def run_episode_td3(Q, env):
    """ Runs an episode for an agent, returns the return."""
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = Q.get_action(state)            
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    return return_

def td3_train(env, agent, max_episodes, std_dev, eps_start, eps_end, eps_decay, start_train_episode, wandb_report):
    all_training_returns = []
    all_actions = []
    all_actions_raw = []
    val_returns = []
    eps = eps_start
    for episode in range(1, max_episodes+1):
        state = env.reset()
        training_return = 0  
        while True:
            if episode < start_train_episode:
                action = np.random.uniform(size=env.env.action_space.shape[0])
                next_state, reward, done, _ = env.step(action)
                training_return += reward
                agent.buffer.add(state, action, reward, next_state, done)
            
            else:
                actor_action = agent.get_action(state)
                # actor_action = expit(actor_action)
                # action_raw = eps * agent.noise.sample(std_dev) + (1. - eps) * actor_action
                # action_raw = actor_action + eps*np.random.uniform(low=-6, high=+6, size=env.num_actions)    #Uniform noise
                action_raw = actor_action + agent.noise.sample(eps*std_dev)
                # action = expit(action_raw)                       # added to include sigmoid
                action = np.clip(action_raw, -1, 1)
                
                next_state, reward, done, _ = env.step(action)
                training_return += reward
                agent.step(state, action, reward, next_state, done)   
                
                all_actions_raw.append(action_raw)
            all_actions.append(action)
        
            # if env.timestep % 24 == 0:
            #     eps = max(eps_end, eps_decay*eps)
                
            if done:
                break
            state = next_state
        if episode >= start_train_episode:     
            eps = max(eps_end, eps_decay*eps)
            # if episode==150: eps=1.
            # eps = 1. - (episode / 180) if episode < 180 else 0.0
        all_training_returns.append(training_return)
        
        # if episode % 1 == 0:
        #     print(f"TD3, episode {episode}, eps: {eps:.2f}, return: {float(training_return):.3e}")
        
        # Calculate return based on current target policy
        if episode % 10 == 0:
            td3_return = run_episode_td3(agent, env)
            val_returns.append(td3_return)
            if wandb_report: wandb.log({'validation_return_TD3': td3_return}, step=episode)
            print(f'TD3, episode {episode}, eps: {eps:.2f}, return: {float(training_return):.1f}, Val return: {float(td3_return):.1f}')
        
        if wandb_report: wandb.log({'training_return_TD3': training_return}, step=episode)
        if episode % 250 == 0:
            torch.save(agent.actor.state_dict(), f'Checkpoint/checkpoint_actor_episode_{episode}.pth')
            torch.save(agent.critic.state_dict(), f'Checkpoint/checkpoint_critic_episode_{episode}.pth')
    all_actions = np.stack(all_actions)
    all_actions_raw = np.stack(all_actions_raw)
    
    return agent, all_training_returns, all_actions, val_returns, all_actions_raw

# ============================================== DEBUGGING CODE ============================================================
# import random
# import torch
# import torch.optim as optim
# # from torch.autograd import Variable
# import torch.nn.functional as F
# from collections import namedtuple, deque
# from DDPG.models import Critic, Actor
# from DDPG.noise import OUActionNoise
# import matplotlib.pyplot as plt
# from env.environment_api import Environment


# class DDPGAgent:
    
#     def __init__(self, env, gamma, tau, buffer_maxlen, batch_size, critic_learning_rate, actor_learning_rate, seed):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.env = env
#         self.obs_dim = env.state_dim
#         self.action_dim = env.action_dim
        
#         # hyperparameters
#         self.num_replay_updates_per_step = 1
#         self.batch_size = batch_size
#         self.gamma = gamma
#         self.tau = tau
        
#         # initialize actor and critic networks
#         self.critic = Critic(self.obs_dim, self.action_dim, seed).to(self.device)
#         self.critic_target = Critic(self.obs_dim, self.action_dim, seed).to(self.device)
        
#         self.actor = Actor(self.obs_dim, self.action_dim, seed).to(self.device)
#         self.actor_target = Actor(self.obs_dim, self.action_dim, seed).to(self.device)
    
#         # Copy critic target parameters
#         for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#             target_param.data.copy_(param.data)
            
#         for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#             target_param.data.copy_(param.data)
        
#         # optimizers
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
#         self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
    
#         self.buffer = ReplayBuffer(buffer_maxlen, batch_size, seed)        
#         self.noise = OUActionNoise()
        
#     def get_action(self, state):
#         # state = Variable(torch.from_numpy(obs).float().unsqueeze(0))
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
#         self.actor.eval()
#         with torch.no_grad():
#             action = self.actor(state)
#         self.actor.train()
        
#         action = action.cpu().data.item()
#         return action
    
#     def step(self, state, action, reward, next_state, done):
#         # Save experience in replay buffer
#         self.buffer.add(state, action, reward, next_state, done)
        
#         # If enough samples are available in buffer, get random subset and learn
#         if len(self.buffer) > self.batch_size:
#             # update the network "num_replay_updates_per_step" times in each step
#             for _ in range(self.num_replay_updates_per_step):
#                 experiences = self.buffer.sample()
#                 self.learn(experiences)
            
            
#     def learn(self, experiences):
#         """Update value parameters using given batch of experience tuples.
#         Params
#         ======
#             experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
#         """
#         states, actions, rewards, next_states, dones = experiences
   
#         curr_Q = self.critic(states, actions)
#         next_actions = self.actor_target(next_states).detach()
#         next_Q = self.critic_target(next_states, next_actions).detach()
#         target_Q = rewards + self.gamma * next_Q * (1 - dones)
        
#         # losses
#         q_loss = F.mse_loss(curr_Q, target_Q)
#         policy_loss = -self.critic(states, self.actor(states).detach()).mean()
        
#           # update actor
#         self.actor_optimizer.zero_grad()
#         policy_loss.backward()
#         self.actor_optimizer.step()
        
#         # update critic
#         self.critic_optimizer.zero_grad()
#         q_loss.backward() 
#         self.critic_optimizer.step()

#         # update target networks 
#         for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
#             target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
#         for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
#             target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    
# class ReplayBuffer:
#     """Fixed-size buffer to store experience tuples."""

#     def __init__(self, buffer_size, batch_size, seed):
#         """Initialize a ReplayBuffer object.
#         Params
#         ======
#             buffer_size (int): maximum size of buffer
#             batch_size (int): size of each training batch
#             seed (int): random seed
#         """
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.buffer = deque(maxlen=buffer_size)  
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
    
#     def add(self, state, action, reward, next_state, done):
#         """Add a new experience to buffer."""
#         e = self.experience(state, action, reward, next_state, done)
#         self.buffer.append(e)
    
#     def sample(self):
#         """Randomly sample a batch of experiences from buffer."""
#         experiences = random.sample(self.buffer, k=self.batch_size)

#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
#         return (states, actions, rewards, next_states, dones)

#     def __len__(self):
#         """Return the current size of internal buffer."""
#         return len(self.buffer)

            
# def find_norm_params(normalize):
#     if normalize:
#         all_states = []
#         state = env.reset()
#         done = False
#         for _ in range(1000):
#             while not done:
#                 all_states.append(state)
#                 action = np.random.uniform()
#                 state, _, done = env.step(action)  
                
#         mu = np.mean(all_states, axis=0)
#         sigma = np.std(all_states, axis=0)
#         sigma[sigma==0] = 1
#     else:
#         mu = np.zeros(env.state_dim) 
#         sigma = np.ones(env.state_dim)        
#     return mu, sigma


# env = Environment()
# mu, sigma = find_norm_params(True)

# Q_ddpg = DDPGAgent(env, gamma=0.8, tau=0, buffer_maxlen=100000, batch_size=8, critic_learning_rate=1e-1, actor_learning_rate=5e-2, seed=1)
# all_training_ddpg, all_actions_ddpg = ddpg_train(env, Q_ddpg, max_episodes=200, mu=mu, sigma=sigma, eps_start=1.0, eps_end=0.01, eps_decay=0.99)

# plt.hist(np.stack(all_actions_ddpg).flatten(), bins=[i/10-0.05 for i in range(12)])