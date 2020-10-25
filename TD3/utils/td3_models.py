import torch
import torch.nn as nn
import torch.nn.functional as F
     
class Actor(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.linear1 = nn.Linear(state_dim, units_fc1) 
        # self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(units_fc1, num_actions)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        # x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, num_actions, seed, units_fc1=256, units_fc2=256, units_fc3=128):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Q1 architecture
        self.linear1a = nn.Linear(state_dim + num_actions, units_fc1)
        self.linear2a = nn.Linear(units_fc1, units_fc2)
        self.linear3a = nn.Linear(units_fc2, units_fc3)
        self.linear4a = nn.Linear(units_fc3, 1)
        
        # Q2 architecture
        self.linear1b = nn.Linear(state_dim + num_actions, units_fc1)
        self.linear2b = nn.Linear(units_fc1, units_fc2)
        self.linear3b = nn.Linear(units_fc2, units_fc3)
        self.linear4b = nn.Linear(units_fc3, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.linear1a(x))
        q1 = F.relu(self.linear2a(q1))
        q1 = F.relu(self.linear3a(q1))
        q1 = self.linear4a(q1)
        
        q2 = F.relu(self.linear1b(x))
        q2 = F.relu(self.linear2b(q2))
        q2 = F.relu(self.linear3b(q2))
        q2 = self.linear4b(q2)
        return q1, q2
    
    def Q1(self, state, action):
        x = torch.cat([state, action], 1)
        
        q1 = F.relu(self.linear1a(x))
        q1 = F.relu(self.linear2a(q1))
        q1 = F.relu(self.linear3a(q1))
        q1 = self.linear4a(q1)
        return q1