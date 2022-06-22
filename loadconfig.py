config_file = './config.json'

import json

with open(config_file) as c:
    config = json.load(c)

sigma = config['noise']['OUActionNoise']['sigma']
theta = config['noise']['OUActionNoise']['theta']
dt = config['noise']['OUActionNoise']['dt']

# print(sigma, theta, dt)