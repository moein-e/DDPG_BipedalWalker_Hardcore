import numpy as np

class GaussianNoise:
    def __init__(self, num_actions, mean=0.0):
        self.name = 'GaussianNoise'
        self.mean = mean
        # self.std_dev = std_dev
        self.size = num_actions
        
    def sample(self, std_dev=2.):
        x = np.random.normal(self.mean, std_dev, self.size)
        return x
    

# Didier: size=ACTION_SPACE_DIM, theta=10 * 0.15, mu=0., sigma=10 * 0.2
# Mine: action_dim=1, mean=0.0, std_dev=1.0, theta=0.15, dt=1e-2, x_initial=None
class OUNoise:   # originally taken from: https://keras.io/examples/rl/ddpg_pendulum/
    def __init__(self, num_actions, mean=0.0, theta=1.5, dt=1e-2, x_initial=None):
        self.name = 'OUNoise'
        self.mean = mean * np.ones(num_actions)
        self.num_actions = num_actions
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def sample(self, std_dev):
        std_dev = std_dev * np.ones(self.num_actions)
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
            
#=============================================
# noise = GaussianNoise()
# n = []
# sigma = 1
# for i in range(120):
#     n.append(noise.sample(sigma))
#     sigma = 0.98 * sigma

# import matplotlib.pyplot as plt
# plt.plot(n)
#============================================

# class OUNoise(object):
#     def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
#         self.mu           = mu
#         self.theta        = theta
#         self.sigma        = max_sigma
#         self.max_sigma    = max_sigma
#         self.min_sigma    = min_sigma
#         self.decay_period = decay_period
#         self.action_dim   = action_space
#         self.low          = 0
#         self.high         = 1
#         self.reset()
        
#     def reset(self):
#         self.state = np.ones(self.action_dim) * self.mu
        
#     def evolve_state(self):
#         x  = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
#         self.state = x + dx
#         return self.state
    
#     def get_action(self, action, t=0):
#         ou_state = self.evolve_state()
#         self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
#         return np.clip(action + ou_state, self.low, self.high)

# https://github.com/openai/gym/blob/master/gym/core.py