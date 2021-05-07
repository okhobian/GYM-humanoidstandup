'''
This file is used to conver generated json histories to excel files
The excel files are used to plot the training logistics
'''

import pandas
import os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
# FILE_PATH = os.path.join(CURRENT_PATH, '2_cartpole', 'logs', 'CartPole-v0', '50000_steps', 'relu_linear_0.001_log.json')
# OUT_FILE = os.path.join(CURRENT_PATH, '2_cartpole', 'logs', 'CartPole-v0', '50000_steps', 'relu_linear_0.001_log.xlsx')

# FILE_PATH = os.path.join(CURRENT_PATH, '4_atari', 'logs', 'Breakout-v0', '500000_steps', 'dqn_Breakout-v0_log.json')
# OUT_FILE = os.path.join(CURRENT_PATH, '4_atari', 'logs', 'Breakout-v0', '500000_steps', 'dqn_Breakout-v0_log.xlsx')

df = pandas.read_json(FILE_PATH)
df.to_excel(OUT_FILE)  
print("file create: {}".format(OUT_FILE))