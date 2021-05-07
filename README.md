# GYM-tryouts

This repo contains a couple of example codes to first tryout some of the GYM reinforcement learnings

## Setup

1. TensorFlow on Windows.
   ```bash
    pip install tensorflow==2.3.1
    ```
2. Keras
   ```bash
    pip install keras-rl2
    ```
3. gym
   ```bash
    pip install gym
    pip install gym[atari]
    ```

## CartPole
2_cartpole/cartpole.py - finished implementation of DQNAgent for cartpole
2_cartpole/logs - contains the logs and weight of training results
## Atari
4_atari/breakout-v0.py - initial attempt to the breakout task
4_atari/logs - contains the logs and weight of training results