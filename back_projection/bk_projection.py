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

#omega x
#phi y
#kappa z

#8192*5378
rows = 8192
cols = 5378
wgs84_a = 6378137
wgs84_f = 1 / 298.257223563
wgs84_b = wgs84_a - wgs84_f * wgs84_a
wgs84_e2 = (wgs84_a * wgs84_a - wgs84_b * wgs84_b) / (wgs84_a * wgs84_a)
F = 1
######config######################
#四元数数据
print('getting quaternion data')
data_att = read_data.read_att()
num_att = data_att.__len__()
att_type = data_att[0].__len__()
#轨迹数据
print('getting GPS data')
t_gps,data_gps,gpsmat = read_data.read_gps()
num_gps = data_gps.__len__()
gps_type = data_gps[0].__len__()
#线阵拍摄时间5378
print('getting timecode')
data_time = read_data.read_time()
num_time = data_time.__len__()
#视向方向数据8192
print('getting look angle')
data_cbr = read_data.read_cbr()
num_cbr = data_cbr.__len__()
cbr_type = data_cbr[0].__len__()
#物点数据
print('grtting control points')
data_XYZ = read_data.get_XYZ()
num_XYZ = data_XYZ.__len__()
XYZ_type = data_XYZ[0].__len__()
#J2000到WGS84旋转矩阵
print('getting rotation angle')
r_time,data_J2W,rmat = read_data.read_J2W()
num_J2W = data_J2W.__len__()
######algorithm###################

###############################################################cal
def cal_a1(phi, omega, kappa):
    a1 = math.cos(phi)*math.cos(kappa)-math.sin(phi)*math.sin(omega)*math.sin(kappa)
    return a1
    # pass

def cal_a2(phi, omega, kappa):
    a2 = -math.cos(phi)*math.sin(kappa)-math.sin(phi)*math.sin(omega)*math.cos(kappa)
    return a2
    # pass

def cal_a3(phi, omega, kappa):
    a3 = -math.sin(phi)*math.cos(omega)
    return a3
    # pass

def cal_b1(phi, omega, kappa):
    b1 = math.cos(omega)*math.sin(kappa)
    return b1
    # pass

def cal_b2(phi, omega, kappa):
    b2 = math.cos(omega)*math.cos(kappa)
    return b2
    # pass

def cal_b3(phi, omega, kappa):
    b3 = -math.sin(omega)
    return b3
    # pass

def cal_c1(phi, omega, kappa):
    c1 = math.sin(phi)*math.cos(kappa)+math.cos(phi)*math.sin(omega)*math.sin(kappa)
    return c1
    # pass

def cal_c2(phi, omega, kappa):
    c2 = -math.sin(phi)*math.sin(kappa)+math.cos(phi)*math.sin(omega)*math.cos(kappa)
    return c2
    # pass

def cal_c3(phi, omega, kappa):
    c3 = math.cos(phi)*math.cos(omega)
    return c3
    # pass


def cal_X(a1, b1, c1, X, Xs, Y, Ys, Z, Zs):
    result = a1*(X-Xs)+b1*(Y-Ys)+c1*(Z-Zs)
    return result
    # pass

def cal_Y(a2, b2, c2, X, Xs, Y, Ys, Z, Zs):
    result = a2*(X-Xs)+b2*(Y-Ys)+c2*(Z-Zs)
    return result
    # pass

def cal_Z(a3, b3, c3, X, Xs, Y, Ys, Z, Zs):
    result = a3*(X-Xs)+b3*(Y-Ys)+c3*(Z-Zs)
    return result
    # pass


def cal_a11(Z_, a1, f, a3, x, x0):
    a11 = (a1*f+a3*x-a3*x0)/Z_
    return a11
    # pass

def cal_a12(Z_, b1, f, b3, x, x0):
    a12 = (b1*f+b3*x-b3*x0)/Z_
    return a12
    # pass

def cal_a13(Z_, c1, f, c3, x, x0):
    a13 = (c1*f+c3*x-c3*x0)/Z_
    return a13
    # pass

def cal_a14(x, x0, y, y0, phi, omega, kappa, f):
    a14 = (y-y0)*math.sin(omega)-((x-x0)*((x-x0)*math.cos(kappa)-(y-y0)*math.sin(kappa))/f+f*math.cos(kappa))*math.cos(omega)
    return a14
    # pass

def cal_a15(x, x0, y, y0, phi, omega, kappa, f):
    a15 = -f*math.sin(kappa)-(x-x0)*((x-x0)*math.sin(kappa)+(y-y0)*math.cos(kappa))/f
    return a15
    # pass

def cal_a16(y, y0):
    a16 = y-y0
    return a16
    # pass

def cal_a21(Z_, a2, f, a3, y, y0):
    a21 = (a2*f+a3*y-a3*y0)/Z_
    return a21
    # pass

def cal_a22(Z_, b2, f, b3, y, y0):
    a22 = (b2*f+b3*y-b3*y0)/Z_
    return a22
    # pass

def cal_a23(Z_, c2, f, c3, y, y0):
    a23 = (c2*f+c3*y-c3*y0)/Z_
    return a23
    # pass

def cal_a24(x, x0, y, y0, phi, omega, kappa, f):
    a24 = -(x-x0)*math.sin(omega)-((y-y0)*((x-x0)*math.cos(kappa)-(y-y0)*math.sin(kappa))/f-f*math.sin(kappa))*math.cos(omega)
    return a24
    # pass

def cal_a25(x, x0, y, y0, phi, omega, kappa, f):
    a25 = -f*math.cos(kappa)-(y-y0)*((x-x0)*math.sin(kappa)+(y-y0)*math.cos(kappa))/f
    return a25
    # pass

def cal_a26(x, x0):
    a26 = -x+x0
    return a26
    # pass

#已解求外方位元素方程
#带入时间返回外方位元素
def cal_gps(time,a0,a1,a2,b0,b1,b2,c0,c1,c2):
    
    X = a0+a1*time+a2*pow(time,2)
    Y = b0+b1*time+b2*pow(time,2)
    Z = c0+c1*time+c2*pow(time,2)
    # O = d0+d1*time+d2*pow(time,2)#omega
    # P = e0+e1*time+e2*pow(time,2)#phi
    # K = f0+f1*time+f2*pow(time,2)#kappa

    return X,Y,Z

#计算旋转矩阵
def cal_para(phi,omega,kappa):
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

#计算像方点
def cal_ptopts(R,X,Y,Z,Xs,Ys,Zs):

    R = R.transpose()
    XX = cal_X(R[0][0],R[1][0],R[2][0],X,Xs,Y,Ys,Z,Zs)
    YY = cal_Y(R[0][1],R[1][1],R[2][1],X,Xs,Y,Ys,Z,Zs)
    ZZ = cal_Z(R[0][2],R[1][2],R[2][2],X,Xs,Y,Ys,Z,Zs)

    x = -F*XX/ZZ
    y = -F*YY/ZZ

    # print(x,y)
    # print('\n')
    # print(XX,YY,ZZ)

    return x,y

#四元数转换为旋转矩阵
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
    #check
    a1 = 1-2*(pow(q2,2)+pow(q3,2))
    a2 = 2*(q1*q2-q3*q4)
    a3 = 2*(q1*q3+q2*q4)
    b1 = 2*(q1*q2+q3*q4)
    b2 = 1-2*(pow(q1,2)+pow(q3,2))
    b3 = 2*(q2*q3-q1*q4)
    c1 = 2*(q1*q3-q2*q4)
    c2 = 2*(q2*q3+q1*q4)
    c3 = 1-2*(pow(q1,2)+pow(q2,2))
    R = np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
    R = R.transpose()
    return R

#判断是旋转矩阵
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#旋转矩阵转欧拉角
def R2A(R) :
    assert(isRotationMatrix(R))

    phi = -math.atan(R[0][2]/R[2][2])
    omega = -math.asin(R[1][2])
    kappa = -math.atan(R[1][0]/R[1][1])
    
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

###############################################################fitting

#找到前后元素
def get_i(time,index,data):
    num = data.__len__()
    # data_type = data[0].__len__()
    sequence = []
    
    for i in range(num):
        dt = data[i][index] - time
        sequence.append(abs(dt))

    dt_min = min(sequence)
    dt_loc = sequence.index(dt_min)
    # print(dt_min,dt_loc)
    dt1 = sequence[dt_loc-1]
    dt2 = sequence[dt_loc+1]

    # print(dt1, dt2)
    if dt1<dt2:
        dt_return = dt1
    else:
        dt_return = dt2
    dt_return_loc = sequence.index(dt_return)
    if dt_return_loc<dt_loc:
        front = dt_return_loc
        back = dt_loc
    else:
        front = dt_loc
        back = dt_return_loc
    # print('front:'+str(front)+'\t'+'back:'+str(back))

    return front, back

#改变数据结构
def get_array(data,index):
    num = data.__len__()
    data_type = data[0].__len__()
    final_array = np.zeros(num)
    for i in range(num):
        final_array[i] = data[i][index]
    return final_array

# 二次函数的标准形式
def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c

# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    return func(params, x) - y

#一次函数的标准形式
def func_sin(params, x):
    k, b = params
    return k * x + b

#一次误差函数
def error_sin(params, x, y):
    return func_sin(params, x) - y

# 对参数求解
def fitting_poly(X, Y):
    p0 = [10, 10, 10]
    Para = leastsq(error, p0, args=(X, Y))
    return Para

#内插
def fitting_poly_sin(X, Y):
    p0 = [10, 10]
    Para = leastsq(error_sin, p0, args=(X, Y))
    return Para

#gps数据内插计算
def interpolation_gps(data):
    '''
    data为data_gps
    data[0]: time
    data[1]: X
    data[2]: Y
    data[3]: Z
    data[4]: O
    data[5]: P
    data[6]: K 
    '''
    time = get_array(data,0)
    x = get_array(data,1)
    y = get_array(data,2)
    z = get_array(data,3)
    # o = get_array(data,4)
    # p = get_array(data,5)
    # k = get_array(data,6)

    #求得a0-f2的数据
    a2,a1,a0 = fitting_poly(time,x)[0]
    b2,b1,b0 = fitting_poly(time,y)[0]
    c2,c1,c0 = fitting_poly(time,z)[0]
    # d2,d1,d0 = fitting_poly(time,o)[0]
    # e2,e1,e0 = fitting_poly(time,p)[0]
    # f2,f1,f0 = fitting_poly(time,k)[0]

    #作图验证部分
    # plt.figure(figsize=(8,6))
    # plt.scatter(time, x, color="green", label="sample data", linewidth=2)

    # X=np.linspace(0,100,100) ##在0-15直接画100个连续点
    # Y=a2*X*X+a1*X+a0 ##函数式
    # plt.plot(X,Y,color="red",label="solution line",linewidth=2)
    # plt.legend() #绘制图例
    # plt.show()

    return (a0,a1,a2,b0,b1,b2,c0,c1,c2)

#四元数内插计算（本体到J2000）
def interpolation_q(data,time):
    front,back = get_i(time,0,data_att)#得到前一项和后一项的行号
    # print(time,front,back)

    t0 = data[front][0]
    t1 = data[back][0]

    q0 = np.array([data[front][1],data[front][2],data[front][3],data[front][4]])
    q1 = np.array([data[back][1],data[back][2],data[back][3],data[back][4]])
    
    q0_ = q0.reshape(1,4)
    q1_ = q1.reshape(4,1)

    q0q1 = float(np.dot(q0_,q1_))
    theta = math.acos(q0q1)
    miu0 = math.sin(theta*(t1-time)/(t1-t0))/math.sin(theta)
    miu1 = math.sin(theta*(time-t0)/(t1-t0))/math.sin(theta)

    qt = miu0*q0+miu1*q1
    # print(qt)

    return qt

#XYZ内插计算
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
    return s.reshape((1,3))

#J2000到WGS84旋转矩阵内插
def InterpRmats(time, Rmats, timeCodes):
    i = 0
    for i in range(len(timeCodes)):
        if time - timeCodes[i] >= 0 and time - timeCodes[i + 1] < 0:
            break
    R1 = Rmats[3 * i:3 * i + 3]
    R2 = Rmats[3 * i + 3:3 * i + 6]
    R = (time - timeCodes[i]) / (timeCodes[i + 1] - timeCodes[i]) * R1 + (timeCodes[i + 1] - time) / (
                timeCodes[i + 1] - timeCodes[i]) * R2
    return R

#角度内插参数
def angle_para():
    angle = []
    for i in range(num_J2W):
        angle.append(R2A(data_J2W[i]))
        # print(angle)
    
    # print(angle)
    phi = get_array(angle,0)
    omega = get_array(angle,1)
    kappa = get_array(angle,2)

    # print(phi)
    # print(omega)
    # print(kappa) 

    k1,b1 = fitting_poly_sin(r_time,phi)[0]
    k2,b2 = fitting_poly_sin(r_time,omega)[0]
    k3,b3 = fitting_poly_sin(r_time,kappa)[0]
    # print(k1,b1,k2,b2,k3,b3)

    return (k1,b1,k2,b2,k3,b3)

#角度内插
def linear_angle(time,r_para):
    phi = r_para[0]*time+r_para[1]
    omega = r_para[2]*time+r_para[3]
    kappa = r_para[4]*time+r_para[5]
    # print(phi,omega,kappa)
    R = cal_para(phi,omega,kappa)
    
    return R

def search_y(dy):
    min_dy = 9999
    old_dy = 9999
    y = 9999
    for i in range(len(data_cbr)):
        check_y = data_cbr[i][0]
        min_dy = abs(check_y-dy)
        if min_dy < old_dy:
            old_dy = min_dy
            y=i
    return y

###############################################################dicho
#二分法迭代求解搜索窗口范围
#搜索出的结果仅为窗口范围，减小搜索区间，并不是最终确认的扫描行
#判断迭代终止的条件：确定该扫描行的外方位元素满足m11*(X-XS)+m12*(Y-YS)+m13*(Z-ZS)=0
def dicho_iter(data):
    num = data.__len__()
    data_type = data[0].__len__()
    # print(num)    
    XX = get_array(data,0)#物方X
    YY = get_array(data,1)#物方Y
    ZZ = get_array(data,2)#物方Z
    NNX = []
    NNY = []
    
    iteration = True
    for i in range(0,num):
        # print('====================================')
        X = XX[i]
        Y = YY[i]
        Z = ZZ[i]
        Ns = 0
        Ne = cols-1
        iteration = True

        while iteration == True:
            # print('iteration')
            N_ = int((Ns+Ne)/2)
            # print(Ns,N_,Ne)
            #check
            # #起点行外方位元素
            XYZs = InterpXYZ(data_time[Ns],gpsmat,t_gps)
            # #中间行外方位元素
            XYZ_ = InterpXYZ(data_time[N_],gpsmat,t_gps)
            # #终点行外方位元素
            XYZe = InterpXYZ(data_time[Ne],gpsmat,t_gps)
            # print(XYZs[0],'\n',XYZ_[0],'\n',XYZe[0])

            #j2000->wgs84矩阵 check
            #Celestial coordinates ( X Y Z) = M x Terrestrial coordinates ( x y z ) 
            RsJ = InterpRmats(data_time[Ns],rmat,r_time)
            R_J = InterpRmats(data_time[N_],rmat,r_time)
            ReJ = InterpRmats(data_time[Ne],rmat,r_time)
            # RsJ = np.linalg.inv(InterpRmats(data_time[Ns],rmat,r_time))
            # R_J = np.linalg.inv(InterpRmats(data_time[N_],rmat,r_time))
            # ReJ = np.linalg.inv(InterpRmats(data_time[Ne],rmat,r_time))
            
            #check
            #四元数内插
            qs = interpolation_q(data_att,data_time[Ns])
            q_ = interpolation_q(data_att,data_time[N_])
            qe = interpolation_q(data_att,data_time[Ne])
            # print(qs,q_,qe)
            #四元数得到的矩阵 本体->J2000
            Rs = q2rotate(qs[0],qs[1],qs[2],qs[3])
            R_ = q2rotate(q_[0],q_[1],q_[2],q_[3])
            Re = q2rotate(qe[0],qe[1],qe[2],qe[3])
            # print(Rs,'\n',R_,'\n',Re)
            # print(RsJ,'\n',R_J,'\n',ReJ)

            #正投影矩阵
            Rots = np.dot(RsJ,Rs)
            Rot_ = np.dot(R_J,R_)
            Rote = np.dot(ReJ,Re)
            # Rots = np.dot(Rs,RsJ)
            # Rot_ = np.dot(R_,R_J)
            # Rote = np.dot(Re,ReJ)

            #求逆 check
            RRs = np.linalg.inv(Rots)
            RR_ = np.linalg.inv(Rot_)
            RRe = np.linalg.inv(Rote)
            # RRs = (Rots)
            # RR_ = (Rot_)
            # RRe = (Rote)

            #计算像点
            # xs,ys = cal_ptopts(RRs,X,Y,Z,XYZs[0,0],XYZs[0,1],XYZs[0,2])
            # x_,y_ = cal_ptopts(RR_,X,Y,Z,XYZ_[0,0],XYZ_[0,1],XYZ_[0,2])
            # xe,ye = cal_ptopts(RRe,X,Y,Z,XYZe[0,0],XYZe[0,1],XYZe[0,2])
            ys,xs = cal_ptopts(RRs,X,Y,Z,XYZs[0,0],XYZs[0,1],XYZs[0,2])
            y_,x_ = cal_ptopts(RR_,X,Y,Z,XYZ_[0,0],XYZ_[0,1],XYZ_[0,2])
            ye,xe = cal_ptopts(RRe,X,Y,Z,XYZe[0,0],XYZe[0,1],XYZe[0,2])
            # print(xs,x_,xe)
            # print(RRs,'\n',RR_,'\n',RRe)

            if xs<0 and x_<0 and xe<0:
                iteration = False
                NNX.append(-1)
                NNY.append(-1)
            elif xs>0 and x_>0 and xe>0:
                iteration = False
                NNX.append(-1)
                NNY.append(-1)
            else:
                if (xs*x_<=0):
                    Ne = N_
                elif (x_*xe<=0):
                    Ns = N_
                else:
                    Ns = int((Ns+N_)/2)
                    Ne = int((Ne+N_)/2)

                if abs(Ne-Ns)<=1:
                    iteration = False
                    NNX.append(int(N_))
                    NNY.append(int(search_y(y_)))
                    
                    print(x_,y_)
                # elif Ns == Ne:
                #     iteration = False
                #     NNX.append(int(N_))
                #     NNY.append(int(search_y(y_)))
                else:
                    iteration = True
                    continue

  
    return NNX,NNY

def XYZ2BLH(data):
    num = data.__len__()
    data_type = data[0].__len__()
    # print(num)    
    XX = get_array(data,0)#物方X
    YY = get_array(data,1)#物方Y
    ZZ = get_array(data,2)#物方Z
    BB = []
    LL = []
    HH = []
    
    iteration = True
    for i in range(0,num):
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
            count = count+1
            W = pow(1-wgs84_e2*pow(math.sin(B),2),0.5)
            N = wgs84_a/W
            H = H+dH
            B = math.atan(Z*(N+H)/(pow(X*X+Y*Y,0.5)*(N*(1-wgs84_e2)+H)))
            L = math.atan(Y/X)
            if Y>0:
                if L<0:
                    L = L+math.pi
            else:
                if L>0:
                    L = L-math.pi
            dH = pow(X*X+Y*Y,0.5)/math.cos(B)-N-H
            if abs(dH)<0.00000001:
                H = H+dH
                iteration = False
            if count>20:
                iteration = False
        
        BB.append(B)
        LL.append(L)
        HH.append(H)
        # print(B,L,H)
    return B,L,H

def BLH2XYZ(data):
    num = data.__len__()
    data_type = data[0].__len__()
    # print(num)    
    BB = get_array(data,0)
    LL = get_array(data,1)
    HH = get_array(data,2)

    data = []

    for i in range(0,num):
        print('====================================')
        B = BB[i]
        L = LL[i]
        H = HH[i]

        W = pow(1-wgs84_e2*pow(math.sin(B),2),0.5)
        N = wgs84_a / W

        X=(N+H)*math.cos(B)*math.cos(L)
        Y=(N+H)*math.cos(B)*math.sin(L)
        Z=(N*(1-wgs84_e2)+H)*math.sin(B)

        data.append([X,Y,Z])

    return data

###############################################################test
#测试航带走向
def data_display(data):
    num = data.__len__()
    data_type = data[0].__len__()

    fig=plt.figure()     
    ax=Axes3D(fig)     
    ax.set_title('point') 
    x = []
    y = []
    z = []
    if(data_type==5):
        print('att')
    elif(data_type==7):
        print('gps')
        for i in range(num):
            x.append(float(data[i][1]))
            y.append(float(data[i][2]))
            z.append(float(data[i][3]))
        
        ax.scatter3D(x,y,z)     
        ax.set_xlabel('x')     
        ax.set_ylabel('y')     
        ax.set_zlabel('z')     
        plt.show()
    elif(data_type==2):
        print('time')
    else:
        print(data_type)

    return 0

#测试plt显示
def plt_test():
    x = np.arange(1,10)
    y = x
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #设置标题
    ax1.set_title('Scatter Plot')
    # #设置X轴标签
    plt.xlabel('X')
    #设置Y轴标签
    plt.ylabel('Y')
    #画散点图
    ax1.scatter(x,y,c = 'r',marker = 'o')
    #设置图标
    plt.legend('x1')
    #显示所画的图
    plt.show()

if __name__ == "__main__":
    num = data_XYZ.__len__()
    # data_xyz = BLH2XYZ(data_XYZ)
    x,y = dicho_iter(data_XYZ)#二分法得到的 行区间范围
    
    for i in range(num):
        print('the feature coordinate===>')
        print('this is the '+str(i+1)+' point')
        print('X:'+str(data_XYZ[i][0]))
        print('Y:'+str(data_XYZ[i][1]))
        print('Z:'+str(data_XYZ[i][2]))
        print('x:'+str(x[i]))
        print('y:'+str(y[i]))
    
    ####################################

    
    # a = interpolation_gps(data_gps)
    # XYZ = InterpXYZ(131862419.00001526,gpsmat,t_gps)
    # X_,Y_,Z_ = cal_gps(131862419.00001526,a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8])#bandon
    # print(XYZ,X_,Y_,Z_)

    # r_para = angle_para()#旋转矩阵线性内插系数
    # R = InterpRmats(131862406,rmat,r_time)
    # R_ = linear_angle(131862406,r_para)
    # print(R,'\n',R_)    
    
    pass
