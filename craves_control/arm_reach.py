import os
import time
import gym
import numpy as np
from gym import spaces
from hardware.usb_cam import camCapture
from pose_estimator import PoseEstimater
from craves_control.hardware import usb_arm
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

class Arm_Reach(gym.Env):
    def __init__(self,
                 setting_file='config/real_arm.json',
                 action_type='continuous',  # 'discrete', 'continuous'
                 ):
        self.yaws = [0, -45, 45]
        self.length = [150, 200, 250]
        self.heights = [40, 20, 20]

        setting = self.load_env_setting(self.get_absdir(setting_file))

        # init hardware
        self.cam = camCapture(0)
        self.cam.start()
        time.sleep(1)
        self.arm_ctl = usb_arm.Arm()
        self.PE = PoseEstimater(setting['model_dir'], 'video')

        # define action
        self.action_type = action_type
        assert self.action_type == 'discrete' or self.action_type == 'continuous'
        if self.action_type == 'discrete':
            self.action_space = spaces.Discrete(len(self.discrete_actions))
        elif self.action_type == 'continuous':
            self.action_space = spaces.Box(low=np.array(self.continous_actions['low']),
                                           high=np.array(self.continous_actions['high']))

        s_high = setting['pose_range']['high'] + setting['goal_range']['high'] + setting['continous_actions'][
            'high']  # arm_pose, target_position, action
        s_low = setting['pose_range']['low'] + setting['goal_range']['low'] + setting['continous_actions']['low']
        self.observation_space = spaces.Box(low=np.array(s_low), high=np.array(s_high))

        self.count_eps = 0
        self.yaws_id = 0
        self.position_id = 0

    def step(self, action):
            info = dict(
                Collision=False,
                Done=False,
                Reward=0.0,
                Action=action,
                Color=None,
                Depth=None,
            )
            self.count_steps += 1
            action = np.squeeze(action)

            if self.action_type == 'discrete':
                cmd = self.discrete_actions[action]
            else:
                action[0] = -action[0]
                cmd = action/5.0
            self.arm_ctl.pwm_ctl(cmd, 0.07)

            # update observation
            img = self.cam.getframe()
            img_rgb = np.array(img[..., ::-1])
            self.pose_last, good = self.PE.pred(img_rgb.copy(), True)
            print('Current pose: ', str(self.pose_last))

            tip_location = self.angle2tip(self.pose_last)
            dis2target = self.get_distance(self.target_location, tip_location, 2)

            self.action_his.append(dis2target)
            state = np.concatenate((np.append(self.pose_last, 0), self.target_pose, action))
            if np.mean(self.action_his[-5:]) < 15 and abs(self.target_location[-1] - tip_location[-1]) < 30:
                reach = True
            else:
                reach = False
            if not good:
                self.count_bad += 1
            else:
                self.count_bad = 0

            if self.count_bad > 5 or self.count_steps > 150 or reach:
                info['Done'] = True
                dt = time.time() - self.start_time
                print('Steps:', str(self.count_steps), 'Times: ', str(dt),
                      'Pose :', str(self.pose_last), 'Distance: ', str(dis2target))
                plt.imshow(img_rgb)
                plt.show()

            return state, 0, info['Done'], info

    def reset(self, ):
        self.PE.reset()
        self.pose_last, good, img_rgb = self.moveto([0, 0, 0, 0], 1000, 5, 0.07)
        time.sleep(3)
        self.PE.reset()
        self.count_steps = 0
        self.target_pose = [self.yaws[self.count_eps/3 % 3],
                            self.length[self.count_eps % 3],
                            self.heights[self.count_eps % 3]]
        self.target_location = self.trz2xyz(self.target_pose)
        print('Start with target pose: ', str(self.target_pose), 'Target: ', str(self.target_location))
        state = np.concatenate((np.append(self.pose_last, 0), self.target_pose, np.zeros(4)))
        self.count_eps += 1
        self.start_time = time.time()
        self.action_his = []
        self.count_bad = 0
        self.factor = 1
        return state

    def close(self):
        # when everything done, release the capture
        self.cam.stop()
        self.arm_ctl.stop()

    def seed(self, seed=None):
        print('fake seed')

    def get_action_size(self):
        return len(self.action)

    def load_env_setting(self, filename):
        f = open(filename)
        file_type = os.path.splitext(filename)[1]
        if file_type == '.json':
            import json
            setting = json.load(f)
        elif file_type == '.yaml':
            import yaml
            setting = yaml.load(f)
        else:
            print('unknown type')

        self.discrete_actions = setting['discrete_actions']
        self.continous_actions = setting['continous_actions']

        return setting

    def bang_bang_controller(self, exp, real):
        error = exp - real
        pos = error > self.dead_area
        neg = error < -self.dead_area
        cmd = pos + neg*-1
        return cmd

    def moveto(self, expected_pose, max_steps=50, th=2, t=0.05):
        error_his = []
        for i in range(max_steps):
            mask = np.array([1, 1, 1, 1])
            img = self.cam.getframe()
            img_rgb = np.array(img[..., ::-1])
            pose, good = self.PE.pred(img_rgb.copy(), True)
            if good:
                error = expected_pose - pose
                mask[np.where(np.abs(error) < th)] = 0  # direction changed, stop control
                ave_error = np.sum(np.abs(error))
                error_his.append(ave_error)
                action = np.clip(error * 0.2, -1, 1)  # simple P controller
                action[0] = -action[0]

                if np.sum(mask) == 0:
                    print('Reach pose: ', str(pose))
                    break
                action = action * mask
                self.arm_ctl.pwm_ctl(action, t)
            else:
                break
        return pose, good, img_rgb

    def angle2tip(self, angles, l1=89.68, l2=113.58, l3=75.0, h1=37.66):

        angles = angles * pi / 180.0
        x0 = - l1 * sin(angles[1]) + l2 * cos(angles[1] + angles[2]) + l3 * cos(angles[1] + angles[2] + angles[3])
        z0 = l1 * cos(angles[1]) + l2 * sin(angles[1] + angles[2]) + l3 * sin(angles[1] + angles[2] + angles[3])

        x = x0 * cos(angles[0])
        y = x0 * sin(angles[0])
        z = z0 + h1
        location = np.array([y, x, z])
        return location

    def get_distance(self, target, current, n=3):
        error = np.array(target[:n]) - np.array(current[:n])
        distance = np.linalg.norm(error)
        return distance

    def xyz2trz(self, xyz):
        theta = np.arctan2(xyz[0], xyz[1])/np.pi*180
        r = np.linalg.norm(xyz[:2])
        z = xyz[2]
        return np.array([theta, r, z])

    def trz2xyz(self, trz):
        x = np.sin(trz[0]/180.0*np.pi)*trz[1]
        y = np.cos(trz[0]/180.0*np.pi)*trz[1]
        z = trz[2]
        return np.array([x, y, z])

    def get_absdir(self, filename):
        import craves_control
        absdir = os.path.join(os.path.dirname(craves_control.__file__), filename)
        return absdir
