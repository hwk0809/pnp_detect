# coding=utf-8
import rospy
import numpy as np
import cv2
import pyquaternion
from pnp_detect.msg import points
import pandas as pd


# MyTime = []
# x = []
# y = []
# z = []
# qw = []
# qx = []
# qy = []
# qz = []
# n = 0
# ======================================================
# 定义坐标变换函数
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
# ros相关函数
# ======================================================
def PNPpoints_InfoCallback(msg):
    rospy.loginfo("Subcribe servo angle Info: angle1:%f  angle2:%f", msg.PNPPoints[4].x, msg.PNPPoints[4].y)
    img_1 = [msg.PNPPoints[0].x, msg.PNPPoints[0].y]
    img_2 = [msg.PNPPoints[3].x, msg.PNPPoints[3].y]
    img_3 = [msg.PNPPoints[2].x, msg.PNPPoints[2].y]
    img_4 = [msg.PNPPoints[1].x, msg.PNPPoints[1].y]
    imgPoints = np.array([img_1, img_2, img_3, img_4], dtype=np.float64)
    time = msg
    # 舵机角度
    UGV_1 = msg.PNPPoints[4].x
    UGV_2 = msg.PNPPoints[4].y

    # ======================================================
    # 坐标系转换部分
    # ======================================================
    # 定义基本参数
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
    img_1 = [msg.PNPPoints[0].x, msg.PNPPoints[0].y]
    img_2 = [msg.PNPPoints[3].x, msg.PNPPoints[3].y]
    img_3 = [msg.PNPPoints[2].x, msg.PNPPoints[2].y]
    img_4 = [msg.PNPPoints[1].x, msg.PNPPoints[1].y]
    # PNP方法求解 机体坐标系系到像素坐标系的坐标变换
    retval, rvec, tvec = cv2.solvePnP(objPoints.reshape(-1, 1, 3), imgPoints.reshape(-1, 1, 2), cameraMatrix,
                                      distCoeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)  # 调用OpenCV库 pnp方法求解旋转向量与平移向量

    R_pnp, _ = cv2.Rodrigues(rvec)  # 旋转向量转化为旋转矩阵
    R_pnp_inverse, t_pnp_inverse = inverse_trans(R_pnp, tvec)

    # print(R_pnp_inverse)
    # print(t_pnp_inverse)

    # 初始时刻相机坐标系 -初始时刻机体坐标系  的坐标变换
    # 这里取vicon系统的坐标原点为机体坐标系的初始位置
    # 相机底座坐标系（底座为前x,左y，上z) -机体初始坐标系（前x,左y，上z）
    # -------------------------rotate-------------------------
    # R_base_to_world = np.array([[0.9873682,-0.14168757,-0.07091308],
    #                             [0.13573134 ,0.98728286, -0.08276202],
    #                             [0.08173762 ,0.07209146 ,0.99404315]])
    # t_base_to_world = np.array([[4920.44665478],[491.45017905],[-364.87261198]])
    # -------------------------static---------------------------------
    # R_base_to_world = np.array([[0.98125596, 0.02913136, 0.19049437],
    #                             [-0.03879804, 0.99813106, 0.04721344],
    #                             [-0.18876295, - 0.05371927, 0.98055229]])
    # t_base_to_world = np.array([[4800.81147428],[450.63177],[-312.26851504]])
    R_base_to_world = np.array([[0.99798953, -0.06042924,  0.01911044],
                                [0.06128012,  0.99698431, -0.04761328],
                                [-0.01617558,  0.04868865,  0.99868302]])
    t_base_to_world = np.array([[4141.26686402],[767.7217012],[-444.68033229]])
    # 相机初始坐标系 - 相机底座坐标系
    R_initCam_to_base, t_initCam_to_base = cal_camera_to_base(0, 0)
    # 相机初始坐标系 - 机体初始坐标系（Vicon坐标系原点）
    R_initCam_to_InitUAV, t_initCam_to_InitUAV = coordinate_trans(R_initCam_to_base, t_initCam_to_base, R_base_to_world,t_base_to_world)

    # 相机坐标系的旋转变换 （初始坐标系位于两个转轴为0度的点，前z，左x，上y）
    theta1 = UGV_1 * np.pi / 180
    theta2 = UGV_2 * np.pi / 180
    R_camera_extrinsic, t_camera_extrinsic = cal_camera_posture(theta1, theta2)

    # 机体坐标系相对于初始状态（世界坐标系）的旋转变换
    # 满足T_pnp = (T_camera)*(T_camera_to_UAV0)* (T_UAV)^-1(T_UAV0_to_UAV)
    R_temp, t_temp = coordinate_trans(R_pnp_inverse, t_pnp_inverse, R_camera_extrinsic, t_camera_extrinsic)
    R_UAV_to_world, t_UAV_to_world = coordinate_trans(R_temp, t_temp, R_initCam_to_InitUAV, t_initCam_to_InitUAV)
    R_world_to_UAV, t_world_to_UAV = inverse_trans(R_UAV_to_world, t_UAV_to_world)

    # 将旋转矩阵转化为四元数
    q_R = list(pyquaternion.Quaternion( matrix=R_world_to_UAV))
    # print("---------------calculate R---------------------")
    # print(R_world_to_UAV)
    # print("---------------calculate t---------------------")
    # print(t_world_to_UAV)

    # 保存成csv格式
    # f = pd.read_csv('rotate_test.csv', index_col=None)
    # time = list(f['time'])
    # x = list(f['x'])
    # y = list(f['y'])
    # z = list(f['z'])
    # qw = list(f['qw'])
    # qx = list(f['qx'])
    # qy = list(f['qy'])
    # qz = list(f['qz'])
    # MyTime.append(msg.header.stamp)
    # x.append(t_world_to_UAV[0][0])
    # y.append(t_world_to_UAV[1][0])
    # z.append(t_world_to_UAV[2][0])
    # qw.append(q_R[0])
    # qx.append(q_R[1])
    # qy.append(q_R[2])
    # qz.append(q_R[3])

# 发布新的话题
    pnp_info_pub = rospy.Publisher('/calculate_info', points, queue_size=10)
    calculate_msg = points()
    calculate_msg.header.stamp = msg.header.stamp
    # calculate_msg. =
    # 初始化learning_topic::Person类型的消息

    calculate_msg.t.x = t_world_to_UAV[0]
    calculate_msg.t.y = t_world_to_UAV[1]
    calculate_msg.t.z = t_world_to_UAV[2]

    calculate_msg.q[0] = q_R[0]
    calculate_msg.q[1] = q_R[1]
    calculate_msg.q[2] = q_R[2]
    calculate_msg.q[3] = q_R[3]

    # 发布消息
    pnp_info_pub.publish(calculate_msg)
    rospy.loginfo("Publsh calculate message")


def pnp_detect_subscriber():
    # ROS节点初始化
    rospy.init_node('detetc', anonymous=True)
    # 创建一个Subscriber，订阅名为/person_info的topic，注册回调函数personInfoCallback
    rospy.Subscriber("/image_pnp_points", points, PNPpoints_InfoCallback)
    # 循环等待回调函数
    rospy.spin()


if __name__ == '__main__':
    pnp_detect_subscriber()
    # data = {'time': MyTime, 'x': x, 'y': y, 'z': z, 'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz}
    # Df = pd.DataFrame(data)
    # Df.to_csv('rotate_test.csv')

