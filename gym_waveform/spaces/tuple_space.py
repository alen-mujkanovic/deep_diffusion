import numpy as np
import gym
from gym_waveform import spaces

class Tuples(gym.spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """
    def __init__(self, spaces, shape):
        gym.spaces.Tuple.__init__(self, spaces)
        self.n = sum([np.prod(s) for s in list(self.size())])
        self.shape = shape

    def size(self):
        sizes = tuple()
        for space in self.spaces:
            if isinstance(space, spaces.Tuples):
                sizes += space.size()
            if isinstance(space, gym.spaces.Discrete):
                sizes += (space.n,)
            elif isinstance(space, gym.spaces.Box):
                sizes += (np.inf,)
            elif isinstance(space, np.ndarray):
                sizes += space.shape + (spaces.Tuples((space[0],)).size(),)
        return sizes

    def sample(self):
        samples = tuple()
        for space in self.spaces:
            if isinstance(space, gym.Space):
                samples += (space.sample(),)
            elif isinstance(space, np.ndarray):
                samples += tuple([i.sample() for i in list(space)]),
        return samples
