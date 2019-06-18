from __future__ import division
import gym
import numpy as np
from collections import deque
from cv2 import resize
from gym.spaces.box import Box
import distutils.version


def create_env(env_id, args):
    if 'RealArm' in env_id:
        from craves_control import arm_reach
        env = arm_reach.Arm_Reach()
    else:
        if 'Unreal' in env_id:
            import gym_unrealcv
        env = gym.make(env_id)
    if args.normalize is True:
        env = NormalizedEnv(env)
    if args.rescale is True:
        env = Rescale(env, args)

    if 'img' in args.obs:
        env = UnrealRescale(env, args)

    env = frame_stack(env, args)  # (n) -> (stack, n) // (c, w, h) -> (stack, c, w, h)

    return env


class Rescale(gym.Wrapper):
    def __init__(self, env, args):
        super(Rescale, self).__init__(env)
        self.mx_d = env.observation_space.high
        self.mn_d = env.observation_space.low
        self.obs_range = self.mx_d - self.mn_d
        self.new_maxd = 1.0
        self.new_mind = -1.0
        self.observation_space = Box(self.new_mind, self.new_maxd, env.observation_space.shape)
        self.args = args
        self.inv_img = np.random.randint(0, 2) == 1 and self.args.inv is True

    def rescale(self, x):
        obs = x.clip(self.mn_d, self.mx_d)
        new_obs = (((obs - self.mn_d) * (self.new_maxd - self.new_mind)
                    ) / self.obs_range) + self.new_mind
        return new_obs

    def reset(self):
        ob = self.env.reset()

        if self.args.inv is True:
            self.inv_img = np.random.randint(0, 2) == 1
            if self.inv_img:
                ob = 255 - ob

        ob = self.rescale(np.float32(ob))
        return ob

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        if self.inv_img:
            ob = 255 - ob
        ob = self.rescale(np.float32(ob))
        return ob, rew, done, info


class UnrealRescale(gym.ObservationWrapper):
    def __init__(self, env, args):
        gym.ObservationWrapper.__init__(self, env)

        self.gray = args.gray
        self.crop = args.crop
        self.input_size = args.input_size
        self.use_gym_10_api = distutils.version.LooseVersion(gym.__version__) >= distutils.version.LooseVersion('0.10.0')
        if self.gray is True:
            if self.use_gym_10_api:
                self.observation_space = Box(-1.0, 1.0, [1, self.input_size, self.input_size], dtype=np.uint8)
            else:
                self.observation_space = Box(-1.0, 1.0, [1, self.input_size, self.input_size])
        else:
            if self.use_gym_10_api:
                self.observation_space = Box(-1.0, 1.0, [3, self.input_size, self.input_size], dtype=np.uint8)
            else:
                self.observation_space = Box(-1.0, 1.0, [3, self.input_size, self.input_size])

    def process_frame_ue(self, frame, size=80, gray=False, crop=False):

        frame = frame.astype(np.float32)

        if crop is True:
            shape = frame.shape
            frame = frame[:shape[0], int(shape[1] / 2 - shape[0] / 2): int(shape[1] / 2 + shape[0] / 2)]
        frame = resize(frame, (size, size))

        if gray is True:
            frame = frame.mean(2)  # color to gray
            frame = np.expand_dims(frame, 0)
        else:
            frame = frame.transpose(2, 0, 1)
        return frame

    def observation(self, observation):
        img = self.process_frame_ue(observation, self.input_size, self.gray, self.crop)
        return img


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class frame_stack(gym.Wrapper):
    def __init__(self, env, args):
        super(frame_stack, self).__init__(env)
        self.stack_frames = args.stack_frames
        self.frames = deque([], maxlen=self.stack_frames)
        # normalize reward
        self.reward_mean = 0
        self.reward_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def reset(self):
        ob = self.env.reset()
        ob = np.float32(ob)
        for _ in range(self.stack_frames):
            self.frames.append(ob)
        return self.observation()

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        ob = np.float32(ob)
        self.frames.append(ob)
        ob = self.observation()
        # rew = self.reward_normalizer(rew)
        return ob, rew, done, info

    def observation(self):
        ob = np.stack(self.frames, axis=0)
        ob = ob.reshape(1, ob.shape[0]*ob.shape[1])
        return ob

    def reward_normalizer(self, reward):
        self.num_steps += 1
        self.reward_mean = self.reward_mean * self.alpha + \
                            reward * (1 - self.alpha)
        self.reward_std = self.reward_std * self.alpha + \
                            reward * (1 - self.alpha)

        unbiased_mean = self.reward_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.reward_std / (1 - pow(self.alpha, self.num_steps))
        return (reward - unbiased_mean) / (unbiased_std + 1e-8)