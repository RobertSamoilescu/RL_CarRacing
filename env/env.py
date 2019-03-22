from gym import Wrapper
import numpy as np
import env.bev as bev
import gym


class CarRacingWrapper(Wrapper):
    # constant number of frames & action
    NUM_FRAMES = 60
    ACTION = np.array([0., 0., 0.])
    ACTION_SPACE = 180

    def __init__(self, env):
        super(CarRacingWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(CarRacingWrapper.ACTION_SPACE)

    def step(self, action):
        steer = (action - 90) / 90.
        action = np.array([steer, 0.1, 0])

        observation, reward, done, info = self.env.step(action)
        observation = bev.from_bird_view(observation)
        observation = np.expand_dims(observation, 2)
        return observation, reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)

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
        steer = np.random.rand(181)
        steer = np.exp(steer)
        steer = steer / steer.sum()
        steer = steer.argmax()

        observation, reward, done, info = env.step(steer)
        print("reward: ",  reward)

        if done:
            break
