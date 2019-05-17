#!/usr/bin/env python3

import argparse
import gym
import time
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from torch_rl.utils.penv import ParallelEnv
from env.env import CarRacingWrapper

import utils

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=10,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=1, # NOT WORKING WITH MORE THAN 1
                    help="number of processes (default: 1)")
parser.add_argument("--argmax", action="store_true", default=True,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="The number of worse episodes to show")

args = parser.parse_args()


def evaluate_model(seed: int, level: int =0):
    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Generate environment

    ENV = "CarRacing-v0"
    envs = []
    for i in range(args.procs):
        env = CarRacingWrapper(gym.make(ENV))
        env.seed(args.seed + 10000 * i)
        env.crt_level = level
        envs.append(env)
    env = ParallelEnv(envs)

    # Define agent

    model_dir = utils.get_model_dir(args.model + "_" + str(seed))
    agent = utils.Agent(ENV, env.observation_space, model_dir, args.argmax, args.procs)
    print("CUDA available: {}\n".format(torch.cuda.is_available()))

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run the agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=agent.device)
    log_episode_num_frames = torch.zeros(args.procs, device=agent.device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, dones, _ = env.step(actions)
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=agent.device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=agent.device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=agent.device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print(
                "- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))

    return logs


def evaluate_models(seeds: list, no_levels: int=3)->dict:
    """
    Evaluate the model given as argument for every level and every seed
    :param seeds: list of integers of model seed
    :param no_levels: number of levels to evaluate on
    :return: dictionary with keys corresponding to levels and values list of rewards
    """
    results = dict()

    for level in range(no_levels):
        print(" * level %d" % (level,))

        results[level] = []

        for seed in seeds:
            print("\t * seed %d" % (seed,))

            logs = evaluate_model(seed, level)
            results[level] += logs["return_per_episode"]

    return results


def plot_results(results: dict):
    """
    box plot, export csv, computea mean and std for every level
    :param results: dict of rewards per level
    :return: dictionary of mean rewards and standard deviation
    """

    # Make box plot

    df = pd.DataFrame.from_dict(results)
    ax = sns.boxplot(data=df, color=".5")
    ax.set(xlabel="level", ylabel="reward")
    plt.show()

    # export as csv

    df.to_csv(args.model + "_results.csv")

    # compute mean and variance

    d = dict()

    d["mean"] = df.mean(axis=0)
    d["var"] = df.var(axis=0)

    return d


if __name__ == "__main__":
    results = evaluate_models([1, 9072])
    plot_results(results)
