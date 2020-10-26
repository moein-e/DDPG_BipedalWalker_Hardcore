import gym
import numpy as np
import torch
from td3_agent import run_episode_td3
from utils.td3_brain import TD3_Agent

gamma = 0.99   
critic_lr = 3e-4    
actor_lr = 1e-4     
batch_size = 128
buffer_size = 1000000
tau = 0.001
update_per_step = 1
actor_std_dev = 0.2
actor_noise_bound = 0.5
actor_update_freq = 2
seed = 5      
env = gym.make("BipedalWalkerHardcore-v3")

Q_td3 = TD3_Agent(env, gamma, tau, buffer_size, batch_size, critic_lr, actor_lr, actor_std_dev, actor_noise_bound, actor_update_freq, seed)

Q_td3.actor.load_state_dict(torch.load('Checkpoint (vocal-mountain-60)/checkpoint_actor_episode_250.pth', map_location=torch.device('cpu')))

# env = gym.wrappers.Monitor(env, "trained_agent/Trained results (run_27_wandb)", force=True)
obs = env.reset()
cum_reward = 0
done = False
for _ in range(500):
# while not done:
  env.render()
  action = Q_td3.get_action(obs)
  obs, reward, done, info = env.step(action)
  cum_reward += reward
env.close()
print(f'cumulative reward = {cum_reward:.2f}')

#==================================================
# n_runs = 10
# td3_return = np.mean([run_episode_td3(Q_td3, env) for _ in range(n_runs)])
# print(f'Average TD3 cumulative reward over {n_runs} runs = {td3_return:.2f}')