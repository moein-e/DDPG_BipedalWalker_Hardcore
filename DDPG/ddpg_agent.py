import numpy as np
import torch
import wandb

def run_episode_ddpg(Q, env):
    """ Runs an episode for an agent, returns the cumulative reward."""
    state = env.reset()
    done = False
    return_ = 0
    while not done:
        action = Q.get_action(state)    
        next_state, reward, done, _ = env.step(action)
        state = next_state
        return_ += reward
    return return_

def ddpg_train(env, agent, max_episodes, std_dev, eps_start, eps_end, eps_decay, wandb_report):
    """ Training the ddpg agent. Applying eps-decay exploration."""
    all_training_returns = []
    all_actions = []
    all_actions_raw = []
    val_returns = []
    eps = eps_start
    q_losses, policy_losses = [], []
    for episode in range(1, max_episodes+1):
        state = env.reset()
        training_return = 0  
        while True:
            #========= Uniform exploration ========
            # explore = np.random.rand() < eps
            # if explore:
            #     action = np.random.uniform(size=env.num_actions)
            #     action_raw = logit(action)
            # else:
            #     action_raw = agent.get_action(state)
            #     action = expit(action_raw)                 # added to include sigmoid
            #=======================================

            actor_action = agent.get_action(state)
            action_raw = actor_action + agent.noise.sample(eps*std_dev)
            action = np.clip(action_raw, -1, 1)
            
            all_actions.append(action)
            all_actions_raw.append(action_raw)
            
            next_state, reward, done, _ = env.step(action)
            training_return += reward
            q_loss, policy_loss = agent.step(state, action, reward, next_state, done)   
            q_losses.append(q_loss)
            policy_losses.append(policy_loss)
                                       
            # if env.timestep % 24 == 0:
            #     eps = max(eps_end, eps_decay*eps)
                
            if done:
                break
            
            state = next_state
                
        eps = max(eps_end, eps_decay*eps)
        # eps = 1. - (episode / 180) if episode < 180 else 0.0
        all_training_returns.append(training_return)
        
        # Calculate return based on current target policy
        if episode % 10 == 0:
            ddpg_return = run_episode_ddpg(agent, env)
            val_returns.append(ddpg_return)
            if wandb_report: wandb.log({'validation_return_DDPG': ddpg_return}, step=episode)
            print(f'episode {episode}, eps: {eps:.2f}, return: {training_return:.1f}, Val return = {ddpg_return:.1f}')
            
        # if episode % 1 == 0:
        #     print(f"DDPG, episode {episode}, eps: {eps:.2f}, return: {training_return:.2f}")
        if wandb_report: wandb.log({'training_return_DDPG': training_return}, step=episode)
        if episode % 250 == 0:
            torch.save(agent.actor.state_dict(), f'Checkpoint/checkpoint_actor_episode_{episode}.pth')
            torch.save(agent.critic.state_dict(), f'Checkpoint/checkpoint_critic_episode_{episode}.pth')
        
    all_actions = np.stack(all_actions)
    all_actions_raw = np.stack(all_actions_raw)
    
    return agent, all_training_returns, all_actions, val_returns, all_actions_raw, q_losses, policy_losses
