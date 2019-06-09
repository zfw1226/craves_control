import cv2
from pose_estimator import PoseEstimater
from hardware.usb_cam import video_capture, camCapture
import numpy as np
from realarm.hardware import usb_arm
import time
import argparse
"""  A simple demo for moving the arm to a target pose  
    Reference pose:
    [0,  0,  0,  0] # init
    [yaw, 69.74, -55.1, -34.7] # 100
    [yaw, 16.2, -44.47, 28.257]  # 150
    [yaw, -16.165, -12.105, 28.27]  # 200
    [yaw, -46.67, 31.777, 14.895]  # 250
"""

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--model-dir', default='real.pth.tar', metavar='MD', help='path to saved models')
parser.add_argument('--raw', dest='raw', action='store_true', help='visualize raw image')
parser.add_argument('--kp', dest='kp', action='store_true', help='visualize image with keypoints')
parser.add_argument('--goal', type=int, default=[0, 0, 0, 0], nargs='+', help='expected pose')

if __name__ == '__main__':
    args = parser.parse_args()
    assert len(args.goal) == 4

    Cam = camCapture(0)  # init camera
    Cam.start()  # start camera
    PE = PoseEstimater(model_dir=args.model_dir, cam_type='video')  # init pose estimator
    arm_ctl = usb_arm.Arm()  # inti arm controller

    # set your expected target pose
    target_pose = args.goal
    time.sleep(1)

    mask = np.array([1, 1, 1, 1])
    img = Cam.getframe()
    img_rgb = np.array(img[..., ::-1])
    pose, good = PE.pred(img_rgb, plot_raw=args.raw, plot_kp=args.kp)
    init_error = target_pose - pose
    while True:
        mask = np.array([1, 1, 1, 1])
        img = Cam.getframe()
        img_rgb = np.array(img[..., ::-1])
        pose, good = PE.pred(img_rgb, plot_raw=args.raw, plot_kp=args.kp)
        if good:
            error = target_pose - pose
            mask[np.where(np.abs(error) < 2)] = 0  # direction changed, stop control
            ave_error = np.sum(np.abs(error))
            action = np.clip(error * 0.2, -1, 1)
            action[0] = -action[0]
            if np.sum(mask) == 0:
                break
            x = 0
            action = action*mask
            arm_ctl.pwm_ctl(action, t=0.05)
    Cam.stop()
    arm_ctl.ctl([0, 0, 0, 0])
    cv2.destroyAllWindows()