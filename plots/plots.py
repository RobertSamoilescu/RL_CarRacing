import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_rewards(min_reward, mean_reward, max_reward):
    plt.plot(min_reward, color='b', label="min reward")
    plt.plot(mean_reward, color='g', label="mean reward")
    plt.plot(max_reward, color='r', label="max_reward")

    plt.xlabel("updates")
    plt.ylabel("reward")
    plt.legend(loc=2)
    plt.title("Rewards")
    plt.show()


def plot_value(value):
    plt.plot(value)
    plt.xlabel("updates")
    plt.ylabel("V")
    plt.title("Value function")
    plt.show()


def plot_log(file: str):
    log = pd.read_csv(file)
    plot_rewards(log.rreturn_min, log.rreturn_mean, log.rreturn_max)
    plot_value(log.value)


def plot_logs(files: list):
    for file in files:
        plot_log(file)


if __name__ == "__main__":
    DIR = "../storage"
    LOGS = "log.csv"
    directories = os.listdir(DIR)

    files = [os.path.join(DIR, directory, LOGS) for directory in directories]
    plot_logs(files)
