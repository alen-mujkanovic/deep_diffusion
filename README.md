# Deep Diffusion


## What is it?

`deep_diffusion` is a reinforcement learning framework for generating asymmetric diffusion encoding gradient waveforms for high-resolution magnetic resonance diffusion imaging. uses varying deep learning agents to learn the diffusion magnetic resonance imaging parameter space and generate optimized spin echo sequences. This allows multi-objective optimization for finding waveforms allowing high encoding efficiency, low echo time (i.e. high SNR), improving motion sensitivity and concomitant field effects. It is written in Python 3 using the deep learning library [Keras](http://keras.io).

## Installation
Install from Github source:

```
git clone https://github.com/alen-mujkanovic/deep_diffusion.git
cd deep_diffusion
python setup.py install
```

## Examples

If you want to run the examples, you'll also have to install:
- **keras-rl** by Matthias Plappert: `pip install keras-rl`
- **gym** by OpenAI: `pip install gym`
- **h5py**: simply run `pip install h5py`

## Citing

If you use `deep_diffusion` in your research, you can cite it as follows:
```bibtex
@misc{mujkanovic2018deepdiffusion,
    author = {Alen Mujkanović},
    title = {deep_diffusion},
    year = {2018},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/alen-mujkanovic/deep_diffusion}},
}
```

## References

1. *keras-rl*, Plappert, 2016 [[GitHub]](https://github.com/keras-rl/keras-rl)
2. [*OpenAI Gym*](arxiv.org/abs/1606.01540), Brockman et al., 2016 [[GitHub]](https://github.com/openai/gym)
3. [*Playing Atari with Deep Reinforcement Learning*](https://arxiv.org/abs/1312.5602), Mnih et al., 2013
4. [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236), Mnih et al., 2015
