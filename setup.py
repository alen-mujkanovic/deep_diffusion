from setuptools import setup
from setuptools import find_packages


setup(name='deep-diffusion',
      version='0.0.1',
      description='Deep Asymmetric Diffusion Encoding',
      author='Alen Mujkanovic',
      author_email='alen.mujkanovic@outlook.com',
      url='https://github.com/alen-mujkanovic/deep_diffusion',
      license='MIT',
      install_requires=['keras>=2.0.7'],
      extras_require={
          'keras-rl': ['keras-rl'],
          'gym': ['gym'],
      },
      packages=find_packages())
