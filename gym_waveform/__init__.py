from gym.envs.registration import registry, register, make, spec
import gym

register(
    id='DiffusionEncoding-v0',
    entry_point='gym_waveform.envs.diffusion_encoding:DiffusionEncodingEnv',
    max_episode_steps=10,
)
