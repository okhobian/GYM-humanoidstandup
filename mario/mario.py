import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

print(SIMPLE_MOVEMENT)
'''
NOOP
RIGHT
RIGHT+A
RIGHT+B
RIGHT+A+B
A
LEFT
'''

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

print(env.observation_space.shape)
print(env.action_space.shape)

done = True
for step in range(100000): # frames
    if done:
        env.reset()
    state, reward, done, info = env.step(env.action_space.sample()) # ramdom actions
    env.render()
env.close()