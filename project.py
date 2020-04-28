

from numpy import *
from math import *


def rotation_matrix(pitch, rol, yaw):  # 输入参数计算旋转矩阵
    R = np.zeros((3, 3))
    R[0, 0] = cos(pitch)*cos(yaw)-sin(pitch)*sin(rol)*sin(yaw)
    R[0, 1] = -cos(pitch)*sin(yaw)-sin(pitch)*sin(rol)*cos(yaw)
    R[0, 2] = -sin(pitch)*cos(rol)
    R[1, 0] = cos(rol)*sin(yaw)
    R[1, 1] = cos(rol)*cos(yaw)
    R[1, 2] = -sin(rol)
    R[2, 0] = sin(pitch)*cos(yaw)+cos(pitch)*sin(rol)*sin(yaw)
    R[2, 1] = -sin(pitch)*sin(yaw)+cos(pitch)*sin(rol)*cos(yaw)
    R[2, 2] = cos(pitch)*cos(rol)
    return R


def q_interpolation(att1, att2, t, flag=0):  # 用读取的四元数数据（包含时间）得到内插完的四元数
    if flag == 0:
        q0 = array([att1[1], att1[2], att1[3], sqrt(1-pow(att1[1], 2)-pow(att1[2], 2)-pow(att1[3], 2))])
        q1 = array([att2[1], att2[2], att2[3], sqrt(1-pow(att2[1], 2)-pow(att2[2], 2)-pow(att2[3], 2))])
    else:
        q0 = array(att1[1:])
        q1 = array(att2[1:])
    t0 = float(att1[0])
    t1 = float(att2[0])
    theta = acos(sum(multiply(q0, q1)))
    y0 = sin(theta*(t1-t)/(t1-t0))/sin(theta)
    y1 = sin(theta*(t-t0)/(t1-t0))/sin(theta)
    return y0*q0+y1*q1


def linear_interpolation(data, t):  # 输入数据组[(t,data1,data2,...),...]
    data = array(data)
    lens = len(data)
    if t < data[0, 0]:
        return [0, 0, 0]
    for i in range(lens):
        if t < data[i, 0]:
            return data[i-1, 1:]+(data[i, 1:]-data[i-1, 1:])\
                   * (data[i, 0]-t)/(data[i, 0]-data[i-1, 0])
    return [0, 0, 0]
