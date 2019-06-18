import torch
import numpy as np
import cv2
from .simple_validate import validate
import os, json, time
from .keypoint2pose import d2tod3
from .hourglass import hg
from .img_loader import get_training_image
from .osutils import Timer

def draw_keypoints(im, keypoints):
    for idx in range(keypoints.shape[1]):
        im = cv2.circle(im, (int(keypoints[0, idx]), int(keypoints[1, idx])), radius=5, color=(255, 0, 0))
    return im


def read_json(file_dir):   
    L, F = [], []
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file)) 
                F.append(file)
    return L, F


def detect_keypoint_2d(model, im, flip=True, scales=[0.75, 1, 1.25], multi_scale=False):
    im = np.asarray(im)
    inputs, meta = get_training_image(im)
    data_dir = 'InvalidInput'
    keypoints = validate(inputs, meta, model, data_dir, flip, scales, multi_scale)
    return keypoints


def init_model(model_dir, num_stacks=2, num_blocks=1, num_classes=17):

    model = hg(
        num_stacks=num_stacks, 
        num_blocks=num_blocks, 
        num_classes=num_classes
    )

    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_dir, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


class PoseEstimater:
    def __init__(self, model_dir='real.pth.tar', cam_type='synthetic'):

        self.model = init_model(self.get_absdir(model_dir))

        with open(self.get_absdir(os.path.join('config', 'camera_parameter.json'))) as f:
            camera_parameter = json.load(f)
        FocalLength = camera_parameter[cam_type]['FocalLength']
        PrincipalPoint = camera_parameter[cam_type]['PrincipalPoint']
        self.camera_intrinstic = np.matrix(
            [[PrincipalPoint[0], FocalLength[0], 0],
             [PrincipalPoint[1], 0, -FocalLength[1]],
             [1, 0, 0]])
        with open(self.get_absdir(os.path.join('config', 'skeleton.json'))) as f:
            self.meta = json.load(f)
        self.pose_history = []
        self.whole_img = False
        self.scales = [0.75, 1.0]
        self.multi_scale = False
        self.flip = False
        self.use_bbox = False
        self.high = np.array([130, 60, 90, 70])
        self.low = np.array([-130, -90, -60, -50])
        self.count_bad = 0

    def pred(self, im_rgb, plot_raw=True, plot_kp=False):
        with Timer("detect"):
            # to balance the speed and accuracy, we remove all of the augmentations(flip, multi-scale)
            keypoints = detect_keypoint_2d(self.model, im_rgb, flip=self.flip, scales=self.scales, multi_scale=self.multi_scale)

            if len(self.pose_history) > 1 and self.count_bad < 5:
                pose_deg, pose_rad, good = d2tod3(keypoints,  self.meta,
                                                  self.camera_intrinstic, init=self.pose_history[-1])
            else:
                pose_deg, pose_rad, good = d2tod3(keypoints, self.meta, self.camera_intrinstic,
                                                  init=None)
            cam_pose_deg = pose_deg[-6:]
            pose_deg = pose_deg[:4]
            if len(self.pose_history) > 1:
                if not self.check_continue(pose_deg):
                    good = False

            if self.check_range(pose_deg):
                good = False

            if good:
                # the good one should be 1.low error, 2.keep continue with last, 3.not out of the range
                self.pose_history.append(pose_rad)
                self.last_pose = pose_deg
                self.count_bad = 0
            else:
                self.count_bad += 1
                print('bad pose')
        if plot_raw:
            im_bgr = im_rgb[..., ::-1]
            cv2.imshow('Raw', im_bgr)

        if plot_kp:
            im_kep = draw_keypoints(im_rgb.copy(), keypoints)
            im_kep = im_kep[..., ::-1]
            cv2.imshow('with keypoint', im_kep)
        if plot_raw or plot_kp:
            cv2.waitKey(1)

        return pose_deg, good

    def reset(self):
        self.pose_history = []
        self.count_bad = 0

    def check_range(self, pose):
        out_max = pose > self.high
        out_min = pose < self.low
        if out_max.sum() + out_min.sum() == 0:
            outrange = False
        else:
            outrange = True
            print('Out of Range: ', str(pose))
        return outrange

    def check_continue(self, pose, th=50):
        distance = np.linalg.norm(pose - self.last_pose)
        if distance > th:
            print('Not Consistant:', distance, pose)
            return False
        else:
            return True

    def get_init(self, mode='cam', num_joints=4):
        def cam_est(ang, scale=500):
            cam_est = scale * np.array([cos(ang[0]) * cos(ang[1]), cos(ang[0]) * sin(ang[1]), sin(ang[0])])
            return cam_est
        import random
        from numpy import pi, sin, cos
        if mode == 'intrinstic':
            x0 = np.random.rand(num_joints + 9)
            if num_joints == 4:
                x0[0] = -x0[0] * 180
                x0[1] = x0[1] * 180
                x0[2:num_joints] = -90 + x0[2:num_joints] * 180
            x0[num_joints] = 70
            x0[num_joints + 1] = -30
            x0[num_joints + 2] = 0
            x0 = x0 * pi / 180
            x0[num_joints + 3:num_joints + 6] = cam_est(x0[num_joints:num_joints + 3], 700)
            x0[num_joints + 6:num_joints + 9] = np.array([983, 521, 1453])

        elif mode == 'cam':
            x0 = np.random.rand(num_joints + 6)
            if num_joints == 4:
                x0[0] = -x0[0] * 180
                x0[1] = x0[1] * 180
                x0[2:num_joints] = -90 + x0[2:num_joints] * 180
            x0[num_joints] = random.randint(10, 60)
            x0[num_joints + 1] = 360 * random.random()
            x0[num_joints + 2] = 0
            x0 = x0 * np.pi / 180
            x0[num_joints + 3:] = cam_est(x0[num_joints:num_joints + 3], 500 + 500 * random.random())

        elif mode == 'arm':
            x0 = np.zeros(num_joints)
            x0 = np.random.rand(num_joints)
            x0[0] = -x0[0] * 180
            x0[1] = x0[1] * 180
            x0[2:] = -90 + x0[2:] * 180
            x0 = x0 * np.pi / 180
        return x0

    def get_absdir(self, filename):
        import craves_control
        absdir = os.path.join(os.path.dirname(craves_control.__file__), filename)
        return absdir


def test_dataset():
    PE = PoseEstimater(cam_type='video')
    folder = 'test_seq'
    imgs_name = os.listdir(folder)
    imgs_name.sort()
    for img_name in imgs_name:
        dir = os.path.join(folder, img_name)
        im = cv2.imread(dir)
        im = np.array(im[..., ::-1])
        pose = PE.pred(im, plot_raw=False, plot_kp=True)
        print(pose)


def test_video():
    from hardware.usb_cam import video_capture, camCapture
    PE = PoseEstimater(cam_type='video')
    Cam = camCapture(0)  # init camera
    Cam.start()  # start camera
    while True:
        img = Cam.getframe()
        img_rgb = np.array(img[..., ::-1])
        pose = PE.pred(img_rgb, plot_raw=False, plot_kp=True)
        print(pose)


if __name__ == '__main__':
    test_video()