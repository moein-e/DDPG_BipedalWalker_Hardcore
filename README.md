# RL: BipedalWalkerHardcore

In this work, **Reinforcement Learning** is used to train a Bipedal Walker (Hardcore) agent to walk and pass through different obstacles. 

Training an agent in the **BipedalWalker Hardcore** [environment](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/) is extremely more challenging than in the original [simple environment](https://gym.openai.com/envs/BipedalWalker-v2/). The original BipedalWalker is trained in another [repository](https://github.com/moein-e/RL_BipedalWalker). 

Deep Deterministic Policy Gradient (**DDPG**) algorithm is implemented for training the agent. In this work, knowledge transfer is employed by initializing the actor neural network parameters to the values obtained from the trained BipedalWalker ([simple version](https://github.com/moein-e/RL_BipedalWalker)). In addition, [Ornstein–Uhlenbeck noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) with decaying standard deviation is used for exploration.

Here is the resulting video after training for 2500 episodes:

<img src="Trained%20Agent/BipedalWalker_Hardcore_Training.gif" width="650">
