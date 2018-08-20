import numpy as np
from gym_waveform import spaces
import gym

from gym.utils import seeding


class DiffusionEncodingEnv(gym.Env):
    """Environment definition
    Diffusion encoding description goes here

    # Action space:
        movInv
        movG

    # Observation space:
        nInv
        G

    # Reward calculation
        duration
        motion
        coco

    """


    def __init__(self):
        """ Initialize the environment
        Set invariants and create action/observation spaces.
        """
        self.dt = 500e-6    # Timestep duration [s]
        self.tEnc = 50e-3   # Encoding duration [s]
        self.Gmax = 150e-3  # Maximum gradient amplitude [T/m]
        self.Smax = 100     # Maximum gradient slew rate [T/m/s]
        self.tRead = 32e-3  # EPI readout duration [s]
        self.tRF = 3.8e-3   # RF pulse duration [s]

        self.dG = self.Smax * self.dt               # Maximum gradient amplitude step [T/m]
        self.nG = int(round(2*self.Gmax/self.dG+1)) # Number of gradient amplitude steps
        self.n = int(round(self.tEnc/self.dt))      # Number of encoding timesteps

        # self.action_space = spaces.Tuple(
        #                             [gym.spaces.Discrete(3)] +
        #                             [gym.spaces.Discrete(3)] * self.n,
        #                             (self.n+1,)
        #                     )
        # self.observation_space = spaces.Tuple(
        #                             [gym.spaces.Discrete(self.n)] +
        #                             [gym.spaces.Discrete(self.nG)] * self.n,
        #                             (self.n+1,)
        #                          )

        self.action_space = spaces.Tuple(
                                np.full(self.n+1, gym.spaces.Discrete(3), dtype=np.object),
                                (self.n+1,)
                            )
        self.observation_space = spaces.Tuple(
                                    np.r_[np.full((1,),gym.spaces.Discrete(self.n)),
                                    np.full(self.n, gym.spaces.Discrete(self.nG), dtype=np.object)],
                                    (self.n+1,)
                                 )

        self.seed()
        self.reset()


    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s)

        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
        """ Resets the state of the environment and returns an initial observation.

        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.step_count = 0
        nInv = int(round(self.n/2))
        self.state = np.array([nInv] + [0]*self.n, dtype=np.int)
        return self.state


    def step(self, action):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).

        # Arguments
            action (object): An action provided by the environment.

        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        assert self.action_space.contains(action+1)

        nInv = self.state[0]
        nInv += action[0]
        self.state[0] = nInv

        G = self.state[1:]
        G += np.array(action[1:])
        self.state[1:] = G

        G = G * self.dG # Convert from discrete units to [T/m]

        if any(G > self.Gmax):
            done = 1
        elif (nInv < 0) or (nInv > self.n):
            done = 1
        else:
            done = 0

        reward = sum(G / self.dG)
        self.step_count += 1

        return self.state, reward, done, {"Steps": self.step_count}


    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)

        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        print(" Steps: ", self.step_count, "\n G: ", self.state)


    def close(self):
        """Perform any necessary cleanup.
        This method is automatically called upon when the program exits.
        """
        print("Succesfully terminated!")
