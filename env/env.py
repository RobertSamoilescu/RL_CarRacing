from gym import Wrapper
import numpy as np
import env.bev as bev

import gym


class CarRacingWrapper(Wrapper):
    # constant number of frames & action
    NUM_FRAMES = 60
    ACTION = np.array([0., 0., 0.])
    STEER_SPACE = 180
    ACC_SPACE = 200

    def __init__(self, env, max_steps=1024):
        super(CarRacingWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(CarRacingWrapper.STEER_SPACE + CarRacingWrapper.ACC_SPACE + 2)
        self.max_steps = max_steps
        self.counter = 0

    def step(self, action):
        # steer and acceleration conversion
        steer = (2. * action[0] - CarRacingWrapper.STEER_SPACE) / CarRacingWrapper.STEER_SPACE
        acc = (2 * action[1] - CarRacingWrapper.ACC_SPACE) / CarRacingWrapper.ACC_SPACE
        action = np.array([steer, acc, 0]) if acc > 0 else np.array([steer, 0, acc])

        # make step
        observation, reward, done, info = self.env.step(action)

        # convert observation from bird eye view to driver view
        observation = bev.from_bird_view(observation)
        observation = np.expand_dims(observation, 2)

        # increment number of steps
        self.counter += 1
        if self.counter > self.max_steps:
        	done = True

        return observation, reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.counter = 0

        # skip first NUM_FRAMES by sending a constant action of doing nothing
        for _ in range(CarRacingWrapper.NUM_FRAMES):
            observation, _, _, _ = self.env.step(CarRacingWrapper.ACTION)

        observation = bev.from_bird_view(observation)
        observation = np.expand_dims(observation, 2)
        return observation


if __name__ == "__main__":
    import cv2

    env = CarRacingWrapper(gym.make("CarRacing-v0"))
    print(env.action_space)
    observation = env.reset()

    while True:
        observation = cv2.resize(observation, (300, 300))
        cv2.imshow("OBS", observation)
        cv2.waitKey(0)

        # random distribution of steer
        steer = np.random.rand(CarRacingWrapper.STEER_SPACE + 1)
        steer = np.exp(steer)
        steer = steer / steer.sum()
        steer = steer.argmax()

        acc = np.random.rand(CarRacingWrapper.ACC_SPACE + 1)
        acc = np.exp(acc)
        acc = acc / acc.sum()
        acc = acc.argmax()

        observation, reward, done, info = env.step((steer, acc))
        print("reward: ",  reward)

        if done:
            break
