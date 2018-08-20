import numpy as np
from rl.core import Processor

class Waveform(Processor):
    """Implementation of abstract processor base class in rl.core

    A processor acts as a coupling mechanism between an `Agent` and its `Env`. This can
    be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlaying
    implementation of the agent or environment.
    """

    def __init__ (self,time_steps,gradient_steps):
        """ Initialize class object """
        self.time_steps = time_steps
        self.gradient_steps = gradient_steps


    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

        # Arguments
            action (int): Action given to the environment

        # Returns
            Processed action given to the environment
        """

        action_array = np.zeros(self.time_steps + 1, dtype=int)
        action_array[action // 3] = action % 3 - 1
        
        return action_array
