import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from utils.td3_models import Critic, Actor
from utils.td3_noise import OUNoise, GaussianNoise

class TD3_Agent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, batch_size, critic_lr, actor_lr, actor_std_dev, actor_noise_bound, actor_update_freq, seed):
        
        # hyperparameters
        self.num_replay_updates_per_step = 1
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor_std_dev = actor_std_dev
        self.actor_noise_bound = actor_noise_bound
        self.actor_update_freq = actor_update_freq
        
        self.t_step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # initialize actor and critic networks
        self.critic = Critic(env.observation_space.shape[0],  env.action_space.shape[0], seed).to(self.device)
        self.critic_target = Critic(env.observation_space.shape[0],  env.action_space.shape[0], seed).to(self.device)
        
        self.actor = Actor(env.observation_space.shape[0],  env.action_space.shape[0], seed).to(self.device)
        self.actor_target = Actor(env.observation_space.shape[0],  env.action_space.shape[0], seed).to(self.device)
        
        # optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
    
        self.buffer = ReplayBuffer(buffer_maxlen, batch_size, seed)        
        self.noise = GaussianNoise(env.action_space.shape[0])
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state)
        action = action.cpu().data.numpy()
        return action
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay buffer
        self.buffer.add(state, action, reward, next_state, done)
        
        # If enough samples are available in buffer, get random subset and learn
        if len(self.buffer) >= self.batch_size:
            # update the network "num_replay_updates_per_step" times in each step
            for _ in range(self.num_replay_updates_per_step):
                experiences = self.buffer.sample()
                self.learn(experiences)        
                
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        self.t_step += 1
        
        states, actions, rewards, next_states, dones = experiences
        
        with torch.no_grad():
			# Select action according to policy and add clipped noise
            actor_noise = (torch.randn_like(actions) * self.actor_std_dev).clamp(-self.actor_noise_bound, self.actor_noise_bound)
            next_actions = (self.actor_target(next_states) + actor_noise).clamp(-1, 1)

			# Compute the target Q value
            next_Q1, next_Q2 = self.critic_target(next_states, next_actions)
            next_Q = torch.min(next_Q1, next_Q2)
            target_Q = rewards + self.gamma * next_Q * (1 - dones)
        
        # Current Q estimates
        curr_Q1, curr_Q2 = self.critic(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(curr_Q1, target_Q) + F.mse_loss(curr_Q2, target_Q)
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()
        
        if self.t_step % self.actor_update_freq == 0:
            # Actor loss
            actor_loss = -self.critic.Q1(states, self.actor(states)).mean()
            
            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            # update target networks 
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
           
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to buffer."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from buffer."""
        experiences = random.sample(self.buffer, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal buffer."""
        return len(self.buffer)