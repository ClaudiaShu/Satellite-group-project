import read_data
import cv2
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# import tensorflow as tf
import scipy
from scipy.optimize import leastsq

# omega x
# phi y
# kappa z

# 8192*5378
rows = 8192
cols = 5378
wgs84_a = 6378137
wgs84_f = 1 / 298.257223563
wgs84_b = wgs84_a - wgs84_f * wgs84_a
wgs84_e2 = (wgs84_a * wgs84_a - wgs84_b * wgs84_b) / (wgs84_a * wgs84_a)
F = 1
######config######################
# 四元数数据
# 影像范围内最大最小高程
min_lat = 52
max_lat = 61

print('getting quaternion data')
data_att = read_data.read_att()
num_att = data_att.__len__()
att_type = data_att[0].__len__()
# 轨迹数据
print('getting GPS data')
t_gps, data_gps, gpsmat = read_data.read_gps()
num_gps = data_gps.__len__()
gps_type = data_gps[0].__len__()
# 线阵拍摄时间5378
print('getting timecode')
data_time = read_data.read_time()
num_time = data_time.__len__()
# 视向方向数据8192
print('getting look angle')
data_cbr = read_data.read_cbr()
num_cbr = data_cbr.__len__()
cbr_type = data_cbr[0].__len__()
# J2000到WGS84旋转矩阵
print('getting rotation angle')
r_time, data_J2W, rmat = read_data.read_J2W()
num_J2W = data_J2W.__len__()
# 相机到卫星本体的姿态关系
data_nad = read_data.read_nad()

######algorithm###################

###############################################################cal
def cal_a1(phi, omega, kappa):
    a1 = math.cos(phi) * math.cos(kappa) - math.sin(phi) * math.sin(omega) * math.sin(kappa)
    return a1
    # pass


def cal_a2(phi, omega, kappa):
    a2 = -math.cos(phi) * math.sin(kappa) - math.sin(phi) * math.sin(omega) * math.cos(kappa)
    return a2
    # pass


def cal_a3(phi, omega, kappa):
    a3 = -math.sin(phi) * math.cos(omega)
    return a3
    # pass


def cal_b1(phi, omega, kappa):
    b1 = math.cos(omega) * math.sin(kappa)
    return b1
    # pass


def cal_b2(phi, omega, kappa):
    b2 = math.cos(omega) * math.cos(kappa)
    return b2
    # pass


def cal_b3(phi, omega, kappa):
    b3 = -math.sin(omega)
    return b3
    # pass


def cal_c1(phi, omega, kappa):
    c1 = math.sin(phi) * math.cos(kappa) + math.cos(phi) * math.sin(omega) * math.sin(kappa)
    return c1
    # pass


def cal_c2(phi, omega, kappa):
    c2 = -math.sin(phi) * math.sin(kappa) + math.cos(phi) * math.sin(omega) * math.cos(kappa)
    return c2
    # pass


def cal_c3(phi, omega, kappa):
    c3 = math.cos(phi) * math.cos(omega)
    return c3
    # pass


def cal_X(a1, b1, c1, X, Xs, Y, Ys, Z, Zs):
    result = a1 * (X - Xs) + b1 * (Y - Ys) + c1 * (Z - Zs)
    return result
    # pass


def cal_Y(a2, b2, c2, X, Xs, Y, Ys, Z, Zs):
    result = a2 * (X - Xs) + b2 * (Y - Ys) + c2 * (Z - Zs)
    return result
    # pass


def cal_Z(a3, b3, c3, X, Xs, Y, Ys, Z, Zs):
    result = a3 * (X - Xs) + b3 * (Y - Ys) + c3 * (Z - Zs)
    return result
    # pass


def cal_a11(Z_, a1, f, a3, x, x0):
    a11 = (a1 * f + a3 * x - a3 * x0) / Z_
    return a11
    # pass


def cal_a12(Z_, b1, f, b3, x, x0):
    a12 = (b1 * f + b3 * x - b3 * x0) / Z_
    return a12
    # pass


def cal_a13(Z_, c1, f, c3, x, x0):
    a13 = (c1 * f + c3 * x - c3 * x0) / Z_
    return a13
    # pass


def cal_a14(x, x0, y, y0, phi, omega, kappa, f):
    a14 = (y - y0) * math.sin(omega) - (
                (x - x0) * ((x - x0) * math.cos(kappa) - (y - y0) * math.sin(kappa)) / f + f * math.cos(
            kappa)) * math.cos(omega)
    return a14
    # pass


def cal_a15(x, x0, y, y0, phi, omega, kappa, f):
    a15 = -f * math.sin(kappa) - (x - x0) * ((x - x0) * math.sin(kappa) + (y - y0) * math.cos(kappa)) / f
    return a15
    # pass


def cal_a16(y, y0):
    a16 = y - y0
    return a16
    # pass


def cal_a21(Z_, a2, f, a3, y, y0):
    a21 = (a2 * f + a3 * y - a3 * y0) / Z_
    return a21
    # pass


def cal_a22(Z_, b2, f, b3, y, y0):
    a22 = (b2 * f + b3 * y - b3 * y0) / Z_
    return a22
    # pass


def cal_a23(Z_, c2, f, c3, y, y0):
    a23 = (c2 * f + c3 * y - c3 * y0) / Z_
    return a23
    # pass


def cal_a24(x, x0, y, y0, phi, omega, kappa, f):
    a24 = -(x - x0) * math.sin(omega) - (
                (y - y0) * ((x - x0) * math.cos(kappa) - (y - y0) * math.sin(kappa)) / f - f * math.sin(
            kappa)) * math.cos(omega)
    return a24
    # pass


def cal_a25(x, x0, y, y0, phi, omega, kappa, f):
    a25 = -f * math.cos(kappa) - (y - y0) * ((x - x0) * math.sin(kappa) + (y - y0) * math.cos(kappa)) / f
    return a25
    # pass


def cal_a26(x, x0):
    a26 = -x + x0
    return a26
    # pass


# 计算旋转矩阵
def cal_para(phi, omega, kappa):
    a1 = cal_a1(phi, omega, kappa)
    a2 = cal_a2(phi, omega, kappa)
    a3 = cal_a3(phi, omega, kappa)

    b1 = cal_b1(phi, omega, kappa)
    b2 = cal_b2(phi, omega, kappa)
    b3 = cal_b3(phi, omega, kappa)

    c1 = cal_c1(phi, omega, kappa)
    c2 = cal_c2(phi, omega, kappa)
    c3 = cal_c3(phi, omega, kappa)

    R = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])

    return R


# 四元数转换为旋转矩阵
def q2rotate(q1, q2, q3, q4):
    '''
    四元数转为旋转矩阵
    本体到J2000
    data为data_att
    data[0]: time
    data[1]: q1
    dara[2]: q2
    dara[3]: q3
    data[4]: q4
    '''

    a1 = 1 - 2 * (pow(q2, 2) + pow(q3, 2))
    a2 = 2 * (q1 * q2 - q3 * q4)
    a3 = 2 * (q1 * q3 + q2 * q4)
    b1 = 2 * (q1 * q2 + q3 * q4)
    b2 = 1 - 2 * (pow(q1, 2) + pow(q3, 2))
    b3 = 2 * (q2 * q3 - q1 * q4)
    c1 = 2 * (q1 * q3 - q2 * q4)
    c2 = 2 * (q2 * q3 + q1 * q4)
    c3 = 1 - 2 * (pow(q1, 2) + pow(q2, 2))
    R = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    R = R.transpose()
    return R


# 旋转矩阵转换为四元数
def rotate2q(R):
    q0 = math.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    q1 = math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) / 2
    if R[1, 2] < R[2, 1]:
        q1 *= -1
    q2 = math.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) / 2
    if R[2, 0] < R[0, 2]:
        q2 *= -1
    q3 = math.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) / 2
    if R[0, 1] < R[1, 0]:
        q3 *= -1
    return np.array([q1, q2, q3, q0])


# 判断是旋转矩阵
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# 旋转矩阵转欧拉角
def R2A(R):
    assert (isRotationMatrix(R))

    phi = -math.atan(R[0][2] / R[2][2])
    omega = -math.asin(R[1][2])
    kappa = -math.atan(R[1][0] / R[1][1])

    # sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    # singular = sy < 1e-6

    # if  not singular :
    #     x = math.atan2(R[2,1] , R[2,2])
    #     y = math.atan2(-R[2,0], sy)
    #     z = math.atan2(R[1,0], R[0,0])
    # else :
    #     x = math.atan2(-R[1,2], R[1,1])
    #     y = math.atan2(-R[2,0], sy)
    #     z = 0

    return np.array([phi, omega, kappa])


# 找到前后元素
def get_i(time, index, data):
    num = data.__len__()
    # data_type = data[0].__len__()
    sequence = []

    for i in range(num):
        dt = data[i][index] - time
        sequence.append(abs(dt))

    dt_min = min(sequence)
    dt_loc = sequence.index(dt_min)
    # print(dt_min,dt_loc)
    dt1 = sequence[dt_loc - 1]
    dt2 = sequence[dt_loc + 1]

    # print(dt1, dt2)
    if dt1 < dt2:
        dt_return = dt1
    else:
        dt_return = dt2
    dt_return_loc = sequence.index(dt_return)
    if dt_return_loc < dt_loc:
        front = dt_return_loc
        back = dt_loc
    else:
        front = dt_loc
        back = dt_return_loc
    # print('front:'+str(front)+'\t'+'back:'+str(back))

    return front, back


# 改变数据结构
def get_array(data, index):
    num = data.__len__()
    data_type = data[0].__len__()
    final_array = np.zeros(num)
    for i in range(num):
        final_array[i] = data[i][index]
    return final_array


# 四元数内插计算（本体到J2000）
def interpolation_q(data, time):
    front, back = get_i(time, 0, data_att)  # 得到前一项和后一项的行号

    t0 = data[front][0]
    t1 = data[back][0]

    q0 = np.array([data[front][1], data[front][2], data[front][3], data[front][4]])
    q1 = np.array([data[back][1], data[back][2], data[back][3], data[back][4]])

    q0_ = q0.reshape(1, 4)
    q1_ = q1.reshape(4, 1)

    q0q1 = float(np.dot(q0_, q1_))
    theta = math.acos(q0q1)
    miu0 = math.sin(theta * (t1 - time) / (t1 - t0)) / math.sin(theta)
    miu1 = math.sin(theta * (time - t0) / (t1 - t0)) / math.sin(theta)

    qt = miu0 * q0 + miu1 * q1
    # print(qt)

    return qt


# 线性内插GPS位置
def InterpXYZ(time, data, timeCodes):
    n = 0
    # timeCodes = get_array(data,0)
    for n in range(len(timeCodes)):
        if time - timeCodes[n] >= 0 and time - timeCodes[n + 1] < 0:
            break

    s = 0
    for j in range(-3, 5):
        p1, p2 = 1, 1
        for i in range(-3, 5):
            if i == j: continue
            p1 *= time - timeCodes[i + n]
            p2 *= timeCodes[j + n] - timeCodes[i + n]
        s += data[j + n] * p1 / p2
    return s.reshape((1, 3))


# 对旋转矩阵进行内插
def InterpRmats(time, Rmats, timeCodes):
    i = 0
    for i in range(len(timeCodes)):
        if time - timeCodes[i] >= 0 and time - timeCodes[i + 1] < 0:
            break
    R1 = Rmats[3 * i:3 * i + 3]
    q0 = rotate2q(R1)
    R2 = Rmats[3 * i + 3:3 * i + 6]
    q1 = rotate2q(R2)

    q0_ = q0.reshape(1, 4)
    q1_ = q1.reshape(4, 1)

    q0q1 = float(np.dot(q0_, q1_))
    theta = math.acos(q0q1)
    miu0 = math.sin(theta * (timeCodes[i + 1] - time) / (timeCodes[i + 1] - timeCodes[i])) / math.sin(theta)
    miu1 = math.sin(theta * (time - timeCodes[i]) / (timeCodes[i + 1] - timeCodes[i])) / math.sin(theta)

    qt = miu0 * q0 + miu1 * q1

    R = q2rotate(qt[0], qt[1], qt[2], qt[3])

    # R2 = (time - timeCodes[i]) / (timeCodes[i + 1] - timeCodes[i]) * R1 + (timeCodes[i + 1] - time) / (
    #             timeCodes[i + 1] - timeCodes[i]) * R2
    return R


# WGS84坐标转J2000？
def XYZ2BLH(data):
    num = data.__len__()
    data_type = data[0].__len__()
    # print(num)
    XX = get_array(data, 0)  # 物方X
    YY = get_array(data, 1)  # 物方Y
    ZZ = get_array(data, 2)  # 物方Z
    BB = []
    LL = []
    HH = []

    iteration = True
    for i in range(0, num):
        print('====================================')
        X = XX[i]
        Y = YY[i]
        Z = ZZ[i]

        H = 0
        dH = 0
        B = 0
        L = 0
        count = 0
        iteration = True

        while iteration == True:
            count = count + 1
            W = pow(1 - wgs84_e2 * pow(math.sin(B), 2), 0.5)
            N = wgs84_a / W
            H = H + dH
            B = math.atan(Z * (N + H) / (pow(X * X + Y * Y, 0.5) * (N * (1 - wgs84_e2) + H)))
            L = math.atan(Y / X)
            if Y > 0:
                if L < 0:
                    L = L + math.pi
            else:
                if L > 0:
                    L = L - math.pi
            dH = pow(X * X + Y * Y, 0.5) / math.cos(B) - N - H
            if abs(dH) < 0.00000001:
                H = H + dH
                iteration = False
            if count > 20:
                iteration = False

        BB.append(B)
        LL.append(L)
        HH.append(H)
        # print(B,L,H)
    return B, L, H


# 生成虚拟控制点 hei wid平面数量 layers高程层数
def virt_grid(hei, wid, layers):
    grid_x = np.zeros((hei, wid), dtype=np.int)
    grid_y = np.zeros((hei, wid), dtype=np.int)
    data = []
    for li in range(hei):
        for lj in range(wid):
            grid_x[li, lj] = int(cols / (hei + 1) * (li+1))
            grid_y[li, lj] = int(rows / (wid + 1) * (lj+1))
            im_time = data_time[grid_x[li, lj]]
            im_cbr = data_cbr[lj]
            im_gps = InterpXYZ(im_time, gpsmat, t_gps)
            R_c2b = cal_para(data_nad[0],data_nad[2],data_nad[4])
            R_j2w = InterpRmats(im_time, rmat, r_time)
            q = interpolation_q(data_att, im_time)
            R_b2j = q2rotate(q[0], q[1], q[2], q[3])
            im_cbr = np.array(im_cbr).transpose()
            u = np.dot(R_j2w, R_b2j)
            u = np.dot(u, R_c2b)
            u = np.dot(u, im_cbr)
            im_gps = np.array(im_gps)
            for lk in range(layers):
                h = (max_lat - min_lat) / (layers + 1) * lk
                a = h + wgs84_a
                b = h + wgs84_b
                A = (u[0] * u[0] + u[1] * u[1]) / a / a + u[2] * u[2] / b / b
                B = 2 * ((im_gps[0, 0] * u[0] + im_gps[0, 1] * u[1]) / a / a + im_gps[0, 2] * u[2] / b / b)
                C = (im_gps[0, 0] * im_gps[0, 0] + im_gps[0, 1] * im_gps[0, 1]) / a / a + im_gps[0, 2] * im_gps[
                    0, 2] / b / b - 1
                uk = (-B - math.sqrt(B * B - 4 * A * C)) / 2 / A
                point_ = [im_gps[0, 0] + uk * u[0], im_gps[0, 1] + uk * u[1], im_gps[0, 2] + uk * u[2],grid_x[li, lj],grid_y[li, lj]]
                data.append(point_)
    data=np.array(data)
    return data

