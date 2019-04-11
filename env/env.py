from gym import Wrapper
import numpy as np
import env.bev as bev
# import bev
import gym


class CarRacingWrapper(Wrapper):
    # constant number of frames & action
    NUM_FRAMES = 60
    ACTION = np.array([0., 0., 0.])
    STEER_SPACE = 180
    ACC_SPACE = 200

    def __init__(self, env, no_stacked_frames=4, max_steps=1024):
        super(CarRacingWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(CarRacingWrapper.STEER_SPACE + CarRacingWrapper.ACC_SPACE + 2)
        self.max_steps = max_steps
        self.counter = 0
        self.no_stacked_frames = no_stacked_frames

    def step(self, action):
        # steer and acceleration conversion
        steer = (2. * action[0].item() - CarRacingWrapper.STEER_SPACE) / CarRacingWrapper.STEER_SPACE
        acc = (2 * action[1].item() - CarRacingWrapper.ACC_SPACE) / CarRacingWrapper.ACC_SPACE
        action = np.array([steer, acc, 0]) if acc > 0 else np.array([steer, 0, acc])

        observations = []
        total_reward = 0

        # make steps
        for _ in range(self.no_stacked_frames):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            # convert observation from bird eye view to driver view
            observation = bev.from_bird_view(observation)
            observations.append(observation)

            # increment number of steps
            self.counter += 1
            if self.counter > self.max_steps:
                done = True
                break

        # check if enough observations
        if len(observations) < self.no_stacked_frames:
            observations = observations + [observations[-1]] * (self.no_stacked_frames - len(observations))

        observations = np.stack(observations, axis=2).transpose(2, 0, 1)
        return observations, total_reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.counter = 0

        # skip first NUM_FRAMES by sending a constant action of doing nothing
        for _ in range(CarRacingWrapper.NUM_FRAMES):
            observation, _, _, _ = self.env.step(CarRacingWrapper.ACTION)

        observations = [bev.from_bird_view(observation)] * self.no_stacked_frames
        observations = np.stack(observations, axis=2).transpose(2, 0, 1)
        return observations

    def seed(self, seed):
        self.env.seed(seed=seed)

    @property
    def track(self):
        return self.env.track


if __name__ == "__main__":
    import cv2

    env = CarRacingWrapper(gym.make("CarRacing-v0"))
    print(env.action_space)
    observation = env.reset()

    while True:
        obs = cv2.resize(observation[0], (300, 300))
        cv2.imshow("OBS", obs)
        cv2.waitKey(0)

        # random distribution of steer
        steer = np.random.rand(CarRacingWrapper.STEER_SPACE + 1)
        steer = np.exp(steer)
        steer = steer / steer.sum()
        steer = steer.argmax()
        steer = 90

        acc = np.random.rand(CarRacingWrapper.ACC_SPACE + 1)
        acc = np.exp(acc)
        acc = acc / acc.sum()
        acc = acc.argmax()
        acc = 120

        observation, reward, done, info = env.step((steer, acc))
        print("reward: ",  reward)
        print(observation[0].shape)

        if done:
            break
