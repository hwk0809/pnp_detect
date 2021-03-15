# -*- coding:utf-8 -*-
"""
作者: hwk
日期: 2021年03月10日
"""
import cv2
import numpy as np
import pyquaternion

# ======================================================
# 输入参数部分：1.四个点像素坐标系坐标  2.舵机两个角度
# ======================================================
# 像素坐标系坐标，四个点的形式分别为[x ,y]，左上顺时针为序
img_1 = [493.57144165, 713.19049072]
img_2 = [532.645812988, 709.5]
img_3 = [533.642883301, 724.261901855]
img_4 = [494.5, 727.354187012]
imgPoints = np.array([img_1, img_2, img_3, img_4], dtype=np.float64)
# 舵机角度
UGV_1 = -2.98828125
UGV_2 = 1.318359375


# ======================================================
# 定义基本函数
# ======================================================
# 四元数转化为旋转矩阵 --四元数顺序为（w,x,y,z)
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


# 坐标转换函数 T = T1*T2
def coordinate_trans(R1=np.eye(3), t1=np.zeros((3, 1)), R2=np.eye(3), t2=np.zeros((3, 1))):
    t_1 = np.array(t1).reshape((-1, 1))
    t_2 = np.array(t2).reshape((-1, 1))
    R = np.matmul(R1, R2)
    t = np.matmul(R1, t_2) + t_1
    return R, t


# 坐标变换位姿求逆 T^-1
def inverse_trans(R=np.eye(3), t=np.zeros((3, 1))):
    t_column = np.array(t).reshape((-1, 1))
    R_new = np.transpose(R)
    t_new = -np.dot(R_new, t_column)
    return R_new, t_new


# 求 相机坐标系 -底座坐标系 的坐标变换
def cal_camera_to_base(theta1, theta2):
    R_link1_to_base = np.array([[np.cos(theta1), -np.sin(theta1), 0],
                                [np.sin(theta1), np.cos(theta1), 0],
                                [0, 0, 1]])  # 第一关节坐标系 - 底座坐标系
    t_link1_to_base = np.array([0, 0, -90]).reshape((-1, 1))
    R_link2_to_link1 = np.array([[np.cos(theta2), 0, np.sin(theta2)],
                                 [0, 1, 0],
                                 [-np.sin(theta2), 0, np.cos(theta2)]])  # 第二关节-第一关节
    t_link2_to_link1 = np.array([-10, 0, -68.05]).reshape((-1, 1))
    R1 = np.array([[0, 0, -1],
                   [0, 1, 0],
                   [1, 0, 0]])  # 绕y轴转-pi/2
    R2 = np.array([[0, -1, 0],
                   [1, 0, 0],
                   [0, 0, 1]])  # 绕z轴转pi/2
    R_temp1, t_temp1 = coordinate_trans(R_link2_to_link1, t_link2_to_link1, R_link1_to_base, t_link1_to_base)
    R_temp2, t_temp2 = coordinate_trans(R1, np.zeros((3, 1)), R_temp1, t_temp1)
    R_camera_to_base, t_camera_to_base = coordinate_trans(R2, np.zeros((3, 1)), R_temp2, t_temp2)
    return R_camera_to_base, t_camera_to_base


# 描述相机坐标系的位姿变换(初始坐标系 - 运动后的坐标系）
def cal_camera_posture(theta1, theta2):
    R_theta, t_theta = cal_camera_to_base(theta1, theta2)
    R_0, t_0 = cal_camera_to_base(0, 0)
    R_0_inverse, t_0_inverse = inverse_trans(R_0, t_0)
    R, t = coordinate_trans(R_theta, t_theta, R_0_inverse, t_0_inverse)
    return R, t


# ======================================================
# 矫正底座参数
# ======================================================
# 定义相机基本参数
K = np.array([[1684.65923868318, 0, 656.480920681900],
              [0, 1684.16379493999, 535.952107137916],
              [0, 0, 1]], dtype=np.float64)  # 相机内参
cameraMatrix = K
distCoeffs = np.array([-0.0622484869016584, 0.116141620134878, 0, 0, 0], dtype=np.float64)  # 径向畸变参数
# 3-D 机体坐标系坐标  --前z，左y，下x
world1 = np.array([10, 55, -150])
world2 = np.array([10, -55, -150])
world3 = np.array([50, -55, -150])
world4 = np.array([50, 55, -150])
objPoints = np.array([world1, world2, world3, world4], dtype=np.float64)
# PNP方法求解 机体坐标系系到像素坐标系的坐标变换
retval, rvec, tvec = cv2.solvePnP(objPoints.reshape(-1, 1, 3), imgPoints.reshape(-1,1,2), cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)  # 调用OpenCV库 pnp方法求解旋转向量与平移向量
R_pnp, _ = cv2.Rodrigues(rvec)  # 旋转向量转化为旋转矩阵
R_pnp_inverse, t_pnp_inverse = inverse_trans(R_pnp, tvec)
print("----------pnp求解的R----------")
print(R_pnp)
print("----------pnp求解的t----------")
print (tvec)
# print(R_pnp_inverse)
# print(t_pnp_inverse)

# Vicon中无人机的位姿 相机坐标系-机体坐标系
quat_standard = [0.724774936, 0.004121771, -0.688612514, 0.022295936]  # Vicon测量 无人机在世界坐标系中的坐标,注意这里是w x y z
R_stadard = quaternion_to_rotation_matrix(quat_standard)
t_standard = np.array([-0.033750105 * 1000, 0.174828059 * 1000, 0.087099887 * 1000])

# 相机坐标系的旋转变换 （初始坐标系位于两个转轴为0度的点，前z，左x，上y）
theta1 = UGV_1 * np.pi / 180
theta2 = UGV_2 * np.pi / 180
R_camera_extrinsic, t_camera_extrinsic = cal_camera_posture(theta1, theta2)

# 相机初始坐标系 - 相机底座坐标系
R_initCam_to_base, t_initCam_to_base = cal_camera_to_base(0, 0)

# 坐标转换
R_camera_extrinsic_inverse, t_camera_extrinsic_inverse = inverse_trans(R_camera_extrinsic, t_camera_extrinsic)
R_initCam_to_base_inverse, t_initCam_to_base_inverse = inverse_trans(R_initCam_to_base, t_initCam_to_base)
R_standard_inverse, t_standard_inverse = inverse_trans(R_stadard, t_standard)
R_temp, t_temp = coordinate_trans(R_initCam_to_base_inverse, t_initCam_to_base_inverse, R_camera_extrinsic_inverse,
                                  t_camera_extrinsic_inverse)
R_temp_1, t_temp_1 = coordinate_trans(R_temp, t_temp, R_pnp, tvec)
R_base_to_world, t_base_to_world = coordinate_trans(R_temp_1, t_temp_1, R_standard_inverse, t_standard_inverse)
print("----------校准的R----------")
print R_base_to_world
print("----------校准的t----------")
print t_base_to_world


