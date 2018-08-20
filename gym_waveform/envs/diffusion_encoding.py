import numpy as np
import gym
from gym.utils import seeding
from gym_waveform import spaces


class DiffusionEncodingEnv(gym.Env):
    """Diffusion Encoding

    """
    def __init__(self):
        self.dt = 500e-6    # Timestep duration [s]
        self.tEnc = 50e-3   # Encoding duration [s]
        self.Gmax = 150e-3  # Maximum gradient amplitude [T/m]
        self.Smax = 100     # Maximum gradient slew rate [T/m/s]
        self.tRead = 32e-3  # EPI readout duration [s]
        self.tRF = 3.8e-3   # RF pulse duration [s]

        self.dG = self.Smax * self.dt               # Maximum gradient amplitude step [T/m]
        self.nG = int(round(2*self.Gmax/self.dG+1)) # Number of gradient amplitude steps
        self.n = int(round(self.tEnc/self.dt))      # Number of encoding timesteps

        self.action_space = spaces.Tuples(
                                    [gym.spaces.Discrete(3)] +
                                    [gym.spaces.Discrete(3)] * self.n,
                                    (self.n+1,)
                            )
        self.observation_space = spaces.Tuples(
                                    [gym.spaces.Discrete(self.n)] +
                                    [gym.spaces.Discrete(self.nG)] * self.n,
                                    (self.n+1,)
                                 )

        self.seed()
        self.reset()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        self.step_count = 0
        self.nInv = int(round(self.n/2))
        self.G = [0] * self.n
        return [self.nInv] + self.G


    def forward(self, observation):
        """Takes the an observation from the environment and returns the action to be taken next.
        If the policy is implemented by a neural network, this corresponds to a forward (inference) pass.

        # Argument
            observation (object): The current observation from the environment.

        # Returns
            The next action to be executed in the environment.
        """
        return action


    def step(self, action):
        print(action)

        stepInv = action[0] - 1
        nInv = self.state[0] + stepInv
        stepG = (np.array(action[1:]) - 1) * self.dG
        G = self.state[1:] + stepG
        self.state = [nInv] + G

        if any(G > self.Gmax):
            done = 1
        elif (nInv < 0) or (nInv > self.n):
            done = 1
        else:
            done = 0

        reward = sum(G)
        self.step_count += 1

        return self.state, reward, done, {"Steps": self.step_count}


    def render(self, mode='human'):
        print("G: ", self.state)
