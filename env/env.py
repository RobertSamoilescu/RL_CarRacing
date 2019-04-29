from gym import Wrapper
import numpy as np
import env.bev as bev
# import bev
import gym


class CarRacingWrapper(Wrapper):
    # constant number of frames & action
    NUM_FRAMES = 60
    ACTION = np.array([0., 0., 0.])

    # action spaces
    STEER_SPACE = 90
    ACC_SPACE = 100

    # domain randomization
    MEAN_X, OFFSET_X = 0, 50 # addition negative or positive, camera ox
    MEAN_Y, OFFSET_Y = 0, 50 # addition negative or positive, camera oy
    MEAN_Z, OFFSET_Z = 25, 5 # addition negative or positive, camera oz
    
    START_TW, OFFSET_TW = 0.5, 1.5  # multiplication or division, track width
    START_TR, OFFSET_TR = 0.5, 1.5  # multiplication or division, turn rate

    START_STEER, OFFSET_STEER = 1, 4 # multiplication or division, steer factor
    START_ACC, OFFSET_ACC = 1, 4     # multiplication or division, acceleration factor
    START_BRK, OFFSET_BRK = 1, 4     # multiplication or division, break factor

    START_FRIC, OFFSET_FRIC = 1, 19 # division, friction factor, by default is set to 10, want to be between 0.1 10


    def __init__(self, env, no_stacked_frames=4, no_past_actions=4, max_steps=1024, no_levels=10, no_steps_per_level=200):
        super(CarRacingWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(2 * CarRacingWrapper.STEER_SPACE + 2 * CarRacingWrapper.ACC_SPACE + 2)
        self.max_steps = max_steps
        self.counter = 0
        self.no_stacked_frames = no_stacked_frames
        self.no_past_actions=no_past_actions

        # level configs
        self.no_levels = no_levels
        self.crt_level = 0. # TODO change back to 0
        self.no_steps_per_level = no_steps_per_level
        self.no_resets = 0

        # list of past actions
        self.action_history = []

        # set observation space
        self.observation_space = (self.observation_space, 3 * no_past_actions) # 3 <= (steer, acc, break)


    def step(self, action):
        # steer and acceleration conversion
        steer = (action[0] - (CarRacingWrapper.STEER_SPACE + 1)) / CarRacingWrapper.STEER_SPACE
        acc = (action[1] - (CarRacingWrapper.ACC_SPACE + 1)) / CarRacingWrapper.ACC_SPACE

        # save origiqqnal action action
        original_action = np.array([steer, acc, 0]) if acc > 0 else np.array([steer, 0, -acc])
        self.action_history.append(original_action)

        # modify steer, acceleration and break according to their factors
        steer = np.clip(self.steer_factor * steer, -1., 1.)
        acc = np.clip(self.acc_factor * acc if acc > 0 else self.brk_factor * acc, -1., 1.)

        # set current action
        action = np.array([steer, acc, 0]) if acc > 0 else np.array([steer, 0, -acc])
    
        observations = []
        total_reward = 0

        # make steps
        for _ in range(self.no_stacked_frames):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward

            # convert observation from bird eye view to driver view
            observation = bev.from_bird_view(observation, offset_x=self.offset_x, offset_y=self.offset_y, offset_z=self.offset_z)
            observation = self.augmentator.augment_image(observation)
            observations.append(observation)

            # increment number of steps
            self.counter += 1
            if self.counter > self.max_steps:
                done = True
                break

        # check if enough observations
        if len(observations) < self.no_stacked_frames:
            observations = observations + [observations[-1]] * (self.no_stacked_frames - len(observations))

        # reshape observations
        observations = np.stack(observations, axis=2).transpose(2, 0, 1)
        past_actions = np.array(self.action_history[-(self.no_past_actions + 1):-1]).reshape(1, -1)

        # construct and return observations
        observations = {"image": observations, "action": past_actions}
        return observations, total_reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        return self.env.render(mode)

    def reset(self, **kwargs):
        # compute camera parameters for current level
        self.offset_x = (2 * np.random.rand() - 1) * CarRacingWrapper.OFFSET_X * self.crt_level/self.no_levels + CarRacingWrapper.MEAN_X
        self.offset_y = (2 * np.random.rand() - 1) * CarRacingWrapper.OFFSET_Y * self.crt_level/self.no_levels + CarRacingWrapper.MEAN_Y
        self.offset_z = (2 * np.random.rand() - 1) * CarRacingWrapper.OFFSET_Z * self.crt_level/self.no_levels + CarRacingWrapper.MEAN_Z

        self.offset_x = int(self.offset_x)
        self.offset_y = int(self.offset_y)
        self.offset_z = int(self.offset_z)

        # compute track parameters for current level
        self.track_width = np.random.rand() * CarRacingWrapper.OFFSET_TW * self.crt_level/self.no_levels + CarRacingWrapper.START_TW
        self.track_radius = np.random.rand() * CarRacingWrapper.OFFSET_TR * self.crt_level/self.no_levels + CarRacingWrapper.START_TR

        # compute steer, acceleration and break level
        self.steer_factor = np.random.rand() * CarRacingWrapper.OFFSET_STEER * self.crt_level/self.no_levels + CarRacingWrapper.START_STEER
        self.steer_factor = 1./self.steer_factor if np.random.randint(2) == 0 else self.steer_factor

        self.acc_factor = np.random.rand() * CarRacingWrapper.OFFSET_ACC * self.crt_level/self.no_levels + CarRacingWrapper.START_ACC
        self.acc_factor = 1./self.acc_factor if np.random.randint(2) == 0 else self.acc_factor

        self.brk_factor = np.random.rand() * CarRacingWrapper.OFFSET_BRK * self.crt_level/self.no_levels + CarRacingWrapper.START_BRK
        self.brk_factor = 1./self.brk_factor if np.random.randint(2) == 0 else self.brk_factor

        # compute friction coeff
        self.fric_factor = np.random.rand() * CarRacingWrapper.OFFSET_FRIC * self.crt_level/self.no_levels + CarRacingWrapper.START_FRIC

        # compute noise amount
        eps = 1e-8
        self.augmentator = bev.get_augmentator(self.crt_level/self.no_levels)

        # print("LEVEL %d" % (self.crt_level,))
        # print("CAMERA_X: %.2f, CAMERA_Y: %.2f, CAMERA_Z: %.2f" % (self.offset_x, self.offset_y, self.offset_z))
        # print("TRACK_WIDTH: %.2f, TRACK_RADIUS: %.2f" % (self.track_width, self.track_radius))
        # print("STEER_FACTOR: %.2f, ACC_FACTOR: %.2f, BRK_FACTOR: %.2f" % (self.steer_factor, self.acc_factor, self.brk_factor))
        # print("FRICTION_FACTOR: %.2f" % (self.fric_factor))

        # set track width, turn rate & friction
        self.env.env.track_width_factor = 1. / self.track_width
        self.env.env.turn_rate_factor = self.track_radius
        self.env.env.friction = 1./self.fric_factor

        # reset env
        self.env.reset()
        self.counter = 0
        self.action_history = [CarRacingWrapper.ACTION] * (self.no_past_actions + 1)

        # increment number of resets & deal with current level
        self.no_resets += 1
        if self.no_resets % self.no_steps_per_level == 0:
            self.crt_level = min(self.crt_level + 1, self.no_levels)
            print("LEVEL %d" % (self.crt_level, ))

        # skip first NUM_FRAMES by sending a constant action of doing nothing
        for _ in range(CarRacingWrapper.NUM_FRAMES):
            observation, _, _, _ = self.env.step(CarRacingWrapper.ACTION)
        
        observation = bev.from_bird_view(observation, offset_x=self.offset_x, offset_y=self.offset_y, offset_z=self.offset_z)
        observation = self.augmentator.augment_image(observation)
        observations = [observation] * self.no_stacked_frames

        # reshape observations
        observations = np.stack(observations, axis=2).transpose(2, 0, 1)
        past_actions = np.array(self.action_history[-(self.no_past_actions+1):-1]).reshape(1, -1)

        # construct and return observations
        observations = {"image": observations, "action": past_actions}
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
        obs = cv2.resize(observation["image"][0], (300, 300))
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
        print(observation["image"][0].shape)

        if done:
            break
