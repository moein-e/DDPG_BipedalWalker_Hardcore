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
        self.critic1 = Critic(env.state_dim, env.num_actions, seed).to(self.device)
        self.critic1_target = Critic(env.state_dim, env.num_actions, seed).to(self.device)
        
        self.critic2 = Critic(env.state_dim, env.num_actions, seed).to(self.device)
        self.critic2_target = Critic(env.state_dim, env.num_actions, seed).to(self.device)
        
        self.actor = Actor(env.state_dim, env.num_actions, seed).to(self.device)
        self.actor_target = Actor(env.state_dim, env.num_actions, seed).to(self.device)
        
        # Copy critic target parameters
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)
        
        # optimizers
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)
    
        self.buffer = ReplayBuffer(buffer_maxlen, batch_size, seed)        
        self.noise = OUNoise(env.num_actions)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state)
        self.actor.train()
        
        action = action.cpu().numpy()
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
        states, actions, rewards, next_states, dones = experiences
        
        actor_noise = torch.normal(torch.zeros(actions.size()), self.actor_std_dev).to(self.device)
        actor_noise = actor_noise.clamp(-self.actor_noise_bound, self.actor_noise_bound)
        next_actions = (self.actor_target(next_states).detach() + actor_noise).clamp(0, 1)
   
        next_Q1 = self.critic1_target(next_states, next_actions).detach()
        next_Q2 = self.critic2_target(next_states, next_actions).detach()
        target_Q = rewards + self.gamma * torch.min(next_Q1, next_Q2) * (1 - dones)
        
        curr_Q1 = self.critic1(states, actions)
        curr_Q2 = self.critic2(states, actions)
        
        # losses
        critic1_loss = F.mse_loss(curr_Q1, target_Q)
        critic2_loss = F.mse_loss(curr_Q2, target_Q)
        
        # update critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward() 
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward() 
        self.critic2_optimizer.step()
        
        if self.t_step % self.actor_update_freq == 0:
            
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
            # update target networks 
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
           
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
                
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
        self.t_step += 1
    
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