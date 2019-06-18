import json
import os
import numpy as np
from numpy import pi, sin, cos
import random
from scipy.optimize import root, least_squares
import cv2


def read_json(file_dir):
    L, F = [], []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
                F.append(file)
    return L, F


def cam_est(ang, scale=500):
    cam_est = scale * np.array([cos(ang[0]) * cos(ang[1]), cos(ang[0]) * sin(ang[1]), sin(ang[0])])
    return cam_est


def make_rotation(pitch, yaw, roll):
    # Convert from degree to radius
    # pitch = pitch / 180.0 * np.pi
    # yaw = yaw / 180.0 * np.pi
    # roll = roll / 180.0 * np.pi
    pitch = pitch
    yaw = yaw  # ???!!!
    roll = roll  # Seems UE4 rotation direction is different
    # from: http://planning.cs.uiuc.edu/node102.html
    ryaw = [
        [-cos(yaw), sin(yaw), 0],
        [-sin(yaw), -cos(yaw), 0],
        [0, 0, 1]
    ]
    rpitch = [
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ]
    rroll = [
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ]
    T = np.matrix(ryaw) * np.matrix(rpitch) * np.matrix(rroll)
    return T


def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img


def Opt(x, num_joints, A, ang, cam, c, d, uv, meta, estimate_cam, estimate_intrinsic, Reprojection):
    if estimate_intrinsic:
        A = np.matrix(
            [[x[num_joints + 6], x[num_joints + 8], 0],
             [x[num_joints + 7], 0, -x[num_joints + 8]],
             [1, 0, 0]])

    num_thetas = num_joints  # first num_theta element of x correspond to theta
    if estimate_cam:
        R = make_rotation(x[num_joints], x[num_joints + 1], x[num_joints + 2]).getI()
        t = -R * np.matrix([x[num_joints + 3], x[num_joints + 4], x[num_joints + 5]]).transpose()
    else:
        R = make_rotation(ang[0], ang[1], ang[2]).getI()
        t = -R * cam
    num_keypoints = d.shape[1]
    mat_t = np.matrix(np.zeros((3, num_keypoints)))
    mat_s = np.matrix(np.zeros((num_keypoints, num_keypoints)))
    s = np.zeros(num_keypoints)
    W = np.matrix(np.zeros((3, num_keypoints)))

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent'])

        if parent == -1:
            mat_t[:, i] = t
            W[:, i] = c + d[:, i]
        elif parent == 0:
            R_joint = np.matrix([
                [cos(x[parent + 1]) * cos(x[0]), -sin(x[0]), -sin(x[parent + 1]) * cos(x[0])],
                [cos(x[parent + 1]) * sin(x[0]), cos(x[0]), -sin(x[parent + 1]) * sin(x[0])],
                [sin(x[parent + 1]), 0, cos(x[parent + 1])]
            ])
            mat_t[:, i] = t
            W[:, i] = c + R_joint * d[:, i]
        else:
            R_joint = np.matrix([
                [cos(x[parent + 1]) * cos(x[0]), -sin(x[0]), -sin(x[parent + 1]) * cos(x[0])],
                [cos(x[parent + 1]) * sin(x[0]), cos(x[0]), -sin(x[parent + 1]) * sin(x[0])],
                [sin(x[parent + 1]), 0, cos(x[parent + 1])]
            ])
            W[:, i] = R_joint * d[:, i]

    right_hand_side = A * R * W + A * mat_t

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent'])
        if parent == 0 or parent == -1:
            s[i] = right_hand_side[2, i]
        else:
            s[i] = right_hand_side[2, i]
            s[i] += s[parent]

    for i in range(num_keypoints):
        parent = int(meta[str(i)]['parent'])
        if parent == 0 or parent == -1:
            mat_s[i, i] = s[i]
        else:
            mat_s[parent, i] = -s[parent]
            mat_s[i, i] = s[i]

    loss = uv[0:2, :] * mat_s - right_hand_side[0:2, :]
    Reprojection[:, :right_hand_side.shape[1]] = right_hand_side[0:2, :] * mat_s.getI()
    loss = np.ravel(loss)
    return loss


def estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic, Reprojection, valid_keypoint_list, meta, cam_intrinstic=None):

    num_joints = meta['num_joints']

    if valid_keypoint_list is None:
        valid_keypoint_list = list(range(len(meta) - 1))

    valid_meta = {'num_joints': meta['num_joints']}
    for i in range(len(valid_keypoint_list)):
        valid_meta[str(i)] = meta[str(valid_keypoint_list[i])]

    valid_uv = uv[:, valid_keypoint_list]

    c = np.matrix(meta['0']['offset']).transpose()

    # rotating joints
    num_keypoints = len(meta) - 1
    d = np.matrix([0, 0, 0]).transpose()
    for i in range(1, num_keypoints):
        d = np.concatenate((d, np.matrix(meta[str(i)]['offset']).transpose()), axis=1)

    valid_num_keypoints = len(valid_meta) - 1
    valid_d = np.matrix([0, 0, 0]).transpose()
    for i in range(1, valid_num_keypoints):
        valid_d = np.concatenate((valid_d, np.matrix(valid_meta[str(i)]['offset']).transpose()), axis=1)

    if estimate_intrinsic:
        A = None
    else:
        A = cam_intrinstic


    res = least_squares(Opt, x0, args=(
    num_joints, A, ang, cam, c, valid_d, valid_uv, valid_meta, estimate_cam, estimate_intrinsic, Reprojection),
                        method='lm', max_nfev=500)

    avg_error = np.average(np.sqrt(np.sum(np.power(valid_uv - Reprojection[:, :valid_uv.shape[1]], 2), axis=0)))

    _ = Opt(res.x, num_joints, A, ang, cam, c, d, uv, meta, estimate_cam, estimate_intrinsic,
            Reprojection)  # Get the reprojection result of the full keypoint list
    return res, Reprojection, avg_error


def heatmap_vis(heatmap):  # for visualization only
    for i in range(len(heatmap)):
        vis = (heatmap[i] * 255).astype(np.uint8)

        cv2.imshow('img', vis)
        cv2.waitKey(0)


def uv_from_heatmap(heatmap, labelmap=None):
    if labelmap is not None:
        heatmap = np.multiply(heatmap, labelmap)

    score = np.amax(heatmap, axis=(1, 2))

    # for i in range(heatmap.shape[0]):
    #     print('joint{}:{}'.format(i, len(np.nonzero(heatmap[i] > 0.7 * score[i])[0])))

    h, w = heatmap.shape[1], heatmap.shape[2]

    heatmap = np.reshape(heatmap, (heatmap.shape[0], h * w))

    uv = np.matrix(np.unravel_index(np.argmax(heatmap, axis=1), (h, w)))[[1, 0], :]

    return uv, score


def get_pred(d2_key, cam_info=None):
    # shape of uv: 2 * 17
    # cam = np.matrix(cam_info[0:3]).transpose()
    # ang = np.array(cam_info[3:6]) * np.pi / 180

    if cam_info != None:
        cam = np.matrix(cam_info[0:3]).transpose()
        ang = np.array(cam_info[3:6]) * np.pi / 180
    else:
        cam = None
        ang = None
    heatmap = None
    # uv = np.matrix(d2_key).transpose()
    uv = d2_key
    score = np.ones(uv.shape[1])

    return uv, score, cam, ang, heatmap

def get_random_init(num_joints=4, mode = 'cam'):
    if mode == 'intrinstic':
        x0 = np.random.rand(num_joints + 9)
        if num_joints == 4:
            x0[0] = -x0[0] * 180
            x0[1] = x0[1] * 180
            x0[2:num_joints] = -90 + x0[2:num_joints] * 180
        x0[num_joints] = 70
        # x0[num_joints+1] = 360*x0[num_joints+1]
        x0[num_joints + 1] = -30
        x0[num_joints + 2] = 0
        x0 = x0 * np.pi / 180
        x0[num_joints + 3:num_joints + 6] = cam_est(x0[num_joints:num_joints + 3], 700)
        x0[num_joints + 6:num_joints + 9] = np.array([983, 521, 1453])
        # print(x0)
        # exit(0)

    elif mode == 'cam':
        x0 = np.random.rand(num_joints + 6)
        if num_joints == 4:
            x0[0] = -x0[0] * 180
            x0[1] = x0[1] * 180
            x0[2:num_joints] = -90 + x0[2:num_joints] * 180
        x0[num_joints] = random.randint(10, 60)
        # x0[num_joints+1] = 360*x0[num_joints+1]
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

def d2tod3(d2_key, meta, cam_intristic=None, cam_info=None, estimate_cam=True, estimate_intrinsic=False, num_joints=4,
           keypoint_list=list(range(17)), init=None, score_th=0.12, kpnum_th=10, error_thres=15):
    good = False
    for i in range(0, 1):
        x = np.zeros(num_joints + 6)
        min_error = -1
        hit = False

        uv, score, cam, ang, heatmap = get_pred(d2_key, cam_info)
        valid_keypoint_list = []
        for j in range(uv.shape[1]):
            if score[j] > score_th and j in keypoint_list:
                valid_keypoint_list.append(j)

        if len(valid_keypoint_list) < kpnum_th:
            print("key point not enough!")
            min_error = -1

        Reprojection = np.zeros((2, uv.shape[1]))

        # optimize equation iteratively
        if init is not None:
            x0 = init
            # use history result for a good init
            res, Reprojection, avg_error = estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic,
                                                             Reprojection, valid_keypoint_list, meta, cam_intristic)
            if avg_error < error_thres:
                min_error = avg_error
                x = res.x
            else:
                x = x0
                print('keypoint is not good enough: ', str(avg_error))

        else:
            # from scratch, try different random init
            for j in range(10):
                x0 = get_random_init()
                res, Reprojection, avg_error = estimate(x0, cam, ang, uv, estimate_cam, estimate_intrinsic,
                                                                 Reprojection, valid_keypoint_list, meta, cam_intristic)
                if avg_error < error_thres:
                    min_error = avg_error
                    x = res.x
                    break
        x_deg = x.copy()
        x_deg[0:7] = x[0:7] * 180 / pi
        x_deg[0] = x_deg[0] + 90
        x_deg[1] = x_deg[1] - 90
        x_deg[3] = x_deg[3] - x_deg[2]
        x_deg[2] = x_deg[2] - x_deg[1]
        while np.max(x_deg[:7]) > 180:
            x_deg[np.where(x_deg[:7] > 180)] -= 360
        while np.min(x_deg[:7]) < -180:
            x_deg[np.where(x_deg[:7] < -180)] += 360
        # print('{}:{}, error:{}'.format(i + 1, hit, min_error))
        if min_error > error_thres or min_error == -1:
            good = False
        else:
            good = True
    return x_deg, x, good


if __name__ == "__main__":
    data_dir = 'C:\\Users\\zuoyi\\Desktop\\Arm\\2dto3d\\20180819\\real_0817'
    pred_dir = os.path.join(data_dir, 'preds')
    meta_dir = 'C:\\Users\\zuoyi\\Desktop\\Arm\\2dto3d\\meta_20180814'
    estimate_cam = True
    estimate_intrinsic = False
    num_joints = 4
    # num_joints = 0
    # cam_type = 'synthetic'
    cam_type = 'video'
    # keypoint_list = [0,1,2,3,4,5,6,9,10,11,12,13]
    # keypoint_list = [0,1,2,3,4,6,9,10,11,12,13]
    keypoint_list = list(range(17))

    FocalLength = camera_parameter['FocalLength']
    PrincipalPoint = camera_parameter['PrincipalPoint']
    cam_intristic = np.matrix(
                    [[PrincipalPoint[0], FocalLength[0], 0],
                    [PrincipalPoint[1], 0, -FocalLength[1]],
                    [1, 0, 0]])
    with open(os.path.join(meta_dir, 'skeleton.json')) as f:
        meta = json.load(f)
    d2tod3(d2_key, meta, cam_intristic)

