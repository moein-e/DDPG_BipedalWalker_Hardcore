import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from td3_agent import td3_train, run_episode_td3
from utils.td3_brain import TD3_Agent
import time
import torch
import wandb

# Hyperparameters ============================
gamma = 0.99   
critic_lr = 1e-4    
actor_lr = 5e-5     
batch_size = 128
buffer_size = 1000000
tau = 0.001
update_per_step = 1
eps_start = 1.0
eps_end = 0.15
eps_decay = 0.998

actor_std_dev = 0.2
actor_noise_bound = 0.5
actor_update_freq = 2
start_train_episode = 1

std_dev = 1.2
seed = 5      
num_episodes = 2000
smoothing_window = 50

# Weight and Biases (wandb) parameters ========
wandb_report = True

if wandb_report:
    wandb.init(project="BipedalWalker-Hardcore")
    config = wandb.config
    
    config.gamma = gamma
    config.critic_lr = critic_lr
    config.actor_lr = actor_lr
    config.batch_size = batch_size
    config.buffer_size = buffer_size
    config.tau = tau
    config.update_per_step = update_per_step
    config.eps_start = eps_start
    config.eps_end = eps_end
    config.eps_decay = eps_decay
    config.std_dev = std_dev
    config.seed = seed
    config.num_episodes = num_episodes
    config.start_train_episode = start_train_episode
    config.smoothing_window = smoothing_window
    config.actor_std_dev = actor_std_dev
    config.actor_noise_bound = actor_noise_bound
    config.actor_update_freq = actor_update_freq
    config.comment = 'Transfer learning - tanh in nn, noise+clip'

#===================
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

env = gym.make("BipedalWalkerHardcore-v3")
env.seed(seed)
state_size = len(env.reset())
num_actions = env.action_space.shape[0]

# Training ####################################################################
Q_td3 = TD3_Agent(env, gamma, tau, buffer_size, batch_size, critic_lr, actor_lr, actor_std_dev, actor_noise_bound, actor_update_freq, seed)

Q_td3.actor.load_state_dict(torch.load('trained_agent/checkpoint_BipedalWalker_actor.pth', map_location=torch.device('cpu')))


if wandb_report: config.actor_nn = str(Q_td3.actor)
if wandb_report: config.critic_nn = str(Q_td3.critic)
if wandb_report: config.noise_type = Q_td3.noise.name 
t0_td3 = time.time()
Q_td3_trained, all_training_td3, all_actions_td3, td3_val_returns, td3_raw_actions= td3_train(env, Q_td3, num_episodes, std_dev, eps_start, eps_end, eps_decay, start_train_episode, wandb_report)
t_td3 = time.time() - t0_td3
all_training_td3 = np.convolve(all_training_td3, np.ones((smoothing_window,))/smoothing_window, mode='valid')

# print(f'Training time: {t_td3/60:.1f} min')

fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111)

ax1.plot(np.arange(len(all_training_td3)), all_training_td3, label='TD3_Train')
ax1.plot(10*np.arange(len(td3_val_returns)), td3_val_returns, label='TD3_Validation')

for item in (ax1.get_xticklabels() + ax1.get_yticklabels()):
    item.set_fontsize(14)    
ax1.set_xlabel('Episode', fontsize=14)
ax1.set_ylabel('Cumulative reward', fontsize=14)
ax1.set_title("Training graph (Smoothed over window size {})".format(smoothing_window))
ax1.legend()
plt.tight_layout()

fig2 = plt.figure(figsize=(10, 6))
ax2 = fig2.add_subplot(111)
ax2.plot(all_actions_td3, '.')
ax2.yaxis.set_major_locator(MultipleLocator(1))
ax2.set_xlabel('Time Step', fontsize=14)
ax2.set_ylabel('Action', fontsize=14)
ax2.set_title('All actions taken during training')
plt.tight_layout()
plt.show()

# =============================== TESTING ===================================
td3_return = np.mean([run_episode_td3(Q_td3_trained, env) for _ in range(10)])
print(f'Average TD3 cumulative reward over 10 runs = {td3_return:.2f}')
if wandb_report: wandb.log({'test_return': td3_return})

# obs = env.reset()
# for _ in range(500):
#   env.render()
#   action = Q_td3_trained.get_action(obs)
#   obs, reward, done, info = env.step(action)
#   if done:
#     obs = env.reset()
# env.close()

# torch.save(Q_td3_trained.actor.state_dict(), 'checkpoint_actor.pth')
# torch.save(Q_td3_trained.critic.state_dict(), 'checkpoint_critic.pth')

# Q_td3_trained.actor.load_state_dict(torch.load('trained_agent/checkpoint1.pth'))

# env = gym.wrappers.Monitor(env, "trained_agent/trained_results", force=True)
# obs = env.reset()
# cum_reward = 0
# done = False
# while not done:
#   env.render()
#   action = Q_td3_trained.get_action(obs)
#   obs, reward, done, info = env.step(action)
#   cum_reward += reward
# env.close()
# print(f'cumulative reward = {cum_reward:.2f}')