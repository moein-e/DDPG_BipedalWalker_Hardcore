import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import torch
import wandb

from ddpg_agent import run_episode_ddpg, ddpg_train
from ddpg_brain import DDPGAgent

# Hyperparameters ============================
gamma = 0.99   
critic_lr = 1e-4    
actor_lr = 5e-5     
batch_size = 256
buffer_size = 1000000
tau = 0.05
eps_start = 1.0
eps_end = 0.2
eps_decay = 0.998
std_dev = 1.3
seed = 5      
num_episodes = 2500
smoothing_window = 50

# Weight and Biases (wandb) parameters ========
wandb_report = False
if wandb_report:
    wandb.init(project="BipedalWalker-Hardcore")
    config = wandb.config
    config.gamma = gamma
    config.critic_lr = critic_lr
    config.actor_lr = actor_lr
    config.batch_size = batch_size
    config.buffer_size = buffer_size
    config.tau = tau
    config.eps_start = eps_start
    config.eps_end = eps_end
    config.eps_decay = eps_decay
    config.std_dev = std_dev
    config.seed = seed
    config.num_episodes = num_episodes
    config.smoothing_window = smoothing_window

#===================
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

env = gym.make("BipedalWalkerHardcore-v3")
env.seed(seed)
state_size = len(env.reset())
num_actions = env.action_space.shape[0]

# Training ####################################################################
Q_ddpg = DDPGAgent(env, gamma, tau, buffer_size, batch_size, critic_lr, actor_lr, seed)

Q_ddpg.actor.load_state_dict(torch.load('Trained Agent/chk_bipedal_simple_actor.pth'))
Q_ddpg.actor.train()

if wandb_report: 
    wandb.watch(Q_ddpg.actor)
    wandb.watch(Q_ddpg.critic)
    config.noise_type = Q_ddpg.noise.name 

Q_ddpg_trained, all_training_ddpg, all_actions_ddpg, ddpg_val_returns, ddpg_raw_actions, q_loss, policy_loss = ddpg_train(env, Q_ddpg, num_episodes, std_dev, eps_start, eps_end, eps_decay, wandb_report)

# Smoothing for plotting purposes
all_training_ddpg = np.convolve(all_training_ddpg, np.ones((smoothing_window,))/smoothing_window, mode='valid')

#=====================
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111)

ax1.plot(np.arange(len(all_training_ddpg)), all_training_ddpg, label='DDPG_Train')
ax1.plot(10*np.arange(len(ddpg_val_returns)), ddpg_val_returns, label='DDPG_Validation')

for item in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(14)    
ax1.set_xlabel('Episode', fontsize=14)
ax1.set_ylabel('Cumulative reward', fontsize=14)
ax1.set_title("Training graph (Smoothed over window size {})".format(smoothing_window))
ax1.legend()
plt.tight_layout()

fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(all_actions_ddpg, '.')
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.set_xlabel('Time Step', fontsize=14)
ax2.set_ylabel('Action', fontsize=14)
ax2.set_title('All actions taken during training')
plt.tight_layout()
plt.show()

fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
ax3.plot(np.convolve(q_loss[batch_size-1:], np.ones((50,))/50, mode='valid'), label='critic_loss')
ax3.set_xlabel('Time Step', fontsize=14)
ax3.set_ylabel('Loss', fontsize=14)
ax3.set_title('Critic Loss (smoothed over window size 50)')
plt.tight_layout()
plt.show()
if wandb_report: wandb.log({'Critic Loss': wandb.Image(fig3)})

fig4 = plt.figure(figsize=(8, 6))
ax4 = fig4.add_subplot(111)
ax4.plot(np.convolve(policy_loss[batch_size-1:], np.ones((50,))/50, mode='valid'), label='actor_loss')
ax4.set_xlabel('Time Step', fontsize=14)
ax4.set_ylabel('Loss', fontsize=14)
ax4.set_title('Actor Loss (smoothed over window size 50)')
plt.tight_layout()
plt.show()
if wandb_report: wandb.log({'Actor Loss': wandb.Image(fig4)})

# =============================== TESTING ===================================
ddpg_return = np.mean([run_episode_ddpg(Q_ddpg_trained, env) for _ in range(10)])
print(f'Average DDPG cumulative reward over 10 runs = {ddpg_return:.2f}')
if wandb_report: wandb.log({'test_return': ddpg_return})