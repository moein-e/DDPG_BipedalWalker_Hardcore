# RL: BipedalWalkerHardcore

The hardcore version of **[BipedalWalker](https://gym.openai.com/envs/BipedalWalkerHardcore-v2/)** is a much harder task than the [original version](https://gym.openai.com/envs/BipedalWalker-v2/). Simple BipedalWalker is trained [here](https://github.com/moein-e/RL_BipedalWalker). 

**DDPG** algorithm is implemented for training the Bipedal Walker (Hardcore) agent. Also, [Ornstein–Uhlenbeck noise](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process) with decaying standard deviation is used for exploration. In addition, actor neural network parameters are initialized by values obtained from the trained BipedalWalker ([simple version](https://github.com/moein-e/RL_BipedalWalker)).

Here is the resulting video:

<img src="Trained%20Agent/BipedalWalker_Hardcore_Training.gif" width="800">
