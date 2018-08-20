import numpy as np
import gym
from gym_waveform import spaces

class Tuple(gym.spaces.Tuple):
    """
    A tuple (i.e., product) of simpler spaces. This class inherits and expands
    from gym.spaces.tuple

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """


    def __init__(self, spaces, shape):
        gym.spaces.Tuple.__init__(self, spaces)
        self.n = sum([np.prod(i) for i in list(self.size())])
        self.shape = shape


    def size(self):
        """ Return number of possible states for each sub(sub)space """
        sizes = tuple()
        for s in self.spaces:
            if isinstance(s, spaces.Tuple):
                sizes += s.size()
            if isinstance(s, gym.spaces.Discrete):
                sizes += (s.n,)
            elif isinstance(s, gym.spaces.Box):
                sizes += (np.inf,)
            elif isinstance(s, np.ndarray):
                sizes += s.shape + (spaces.Tuple((s[0],)).size(),)
        return sizes


    def sample(self):
        """ Uniformly randomly sample a random element of this space """
        samples = tuple()
        for s in self.spaces:
            if isinstance(s, spaces.Tuple):
                samples += (s.sample(),)
            if isinstance(s, gym.Space):
                samples += (s.sample(),)
            elif isinstance(s, np.ndarray):
                samples += tuple([i.sample() for i in list(s)]),
        return samples


    def contains(self, x):
        """ Return boolean specifying if x is a valid member of this space """
        i = 0
        for s in self.spaces:
            if s.contains(x[i]):
                i += 1
            else:
                return False
        return True
