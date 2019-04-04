#!/usr/bin/environment python3

import argparse
import gym
import time
import cv2
from environment.env import CarRacingWrapper


try:
    import gym_minigrid
except ImportError:
    pass

import utils

# Parse arguments

parser = argparse.ArgumentParser()
# parser.add_argument("--environment", required=True,
#                     help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent")
args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

# Generate environment
ENV = "CarRacing-v0"
env = CarRacingWrapper(gym.make(ENV))
env.seed(args.seed)
for _ in range(args.shift):
    env.reset()

# Define agent

model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(ENV, env.observation_space, model_dir, args.argmax)

# Run the agent

done = True

while True:
    if done:
        obs = env.reset()

    time.sleep(args.pause)
    renderer = env.render()

    # show obs
    cv2.imshow("OBS", obs)
    cv2.waitKey(1)

    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(action)

    agent.analyze_feedback(reward, done)

    # if renderer.window is None:
    #     break
