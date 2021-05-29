# coding: UTF-8

import cv2
import numpy as np


def quaternion_to_rotation_matrix(quat):
    q1 = np.array(quat)
    q = q1.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2]]],
        dtype=q.dtype)
    return rot_matrix


# camera intrinsic
K = np.array([[608.763, 0, 638.967],
              [0, 608.523, 363.529],
              [0, 0, 1]], dtype=np.float64)  # 相机内参
cameraMatrix = K
distCoeffs = np.array([0.465018, -2.39, 0.000794036, 1.23308e-06, 1.33069], dtype=np.float64)  # 径向畸变参数

# image point
# img_1 = [493.57144165, 713.19049072]
# img_2 = [532.645812988, 709.5]
# img_3 = [533.642883301, 724.261901855]
# img_4 = [494.5, 727.354187012]
# imgPoints = np.array([img_1, img_2, img_3, img_4], dtype=np.float64)

# world point
world1 = np.array([10, 55, -150])
world2 = np.array([10, -55, -150])
world3 = np.array([50, -55, -150])
world4 = np.array([50, 55, -150])
objPoints = np.array([world1, world2, world3, world4], dtype=np.float64)

# pnp method
retval, rvec, tvec = cv2.solvePnP(objPoints.reshape(-1, 1, 3), imgPoints.reshape(-1,1,2), cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)  # 调用OpenCV库 pnp方法求解旋转向量与平移向量
R_pnp, _ = cv2.Rodrigues(rvec)  # 旋转向量转化为旋转矩阵