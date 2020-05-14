import numpy as np
import cv2
# from libtiff import TIFF


# get att check
def read_att(filename="./data2/DX_ZY3_NAD_att.txt"):
    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(6):
            f.readline()
        data_num = f.readline().split(' ')[2]
        data_num = int(data_num)  # 401

        data = []

        for i in range(data_num):
            for j in range(2):
                f.readline()

            timeCode = float((f.readline().lstrip()).split(' ')[2])
            for j in range(7):
                f.readline()

            q1 = float((f.readline().lstrip()).split(' ')[2])
            q2 = float((f.readline().lstrip()).split(' ')[2])
            q3 = float((f.readline().lstrip()).split(' ')[2])
            q4 = float((f.readline().lstrip()).split(' ')[2])
            data.append([timeCode, q1, q2, q3, q4])

            f.readline()
    # print(data[0])
    return data


# get gps check
def read_gps(filename="./data2/DX_ZY3_NAD_gps.txt"):

    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(4):
            f.readline()
        data_num = f.readline().split(' ')[2]  # 101
        data_num = int(data_num)
        data = []

        for i in range(data_num):
            for j in range(2):
                f.readline()

            timeCode = float((f.readline().lstrip()).split(' ')[2])
            f.readline()

            PX = float((f.readline().lstrip()).split(' ')[2])
            PY = float((f.readline().lstrip()).split(' ')[2])
            PZ = float((f.readline().lstrip()).split(' ')[2])
            VX = float((f.readline().lstrip()).split(' ')[2])
            VY = float((f.readline().lstrip()).split(' ')[2])
            VZ = float((f.readline().lstrip()).split(' ')[2])

            data.append([timeCode, PX, PY, PZ])

            f.readline()

    gpsmat = np.zeros((data_num, 3))
    T = np.zeros(data_num)
    # gpsmat = data[0:data_num,1:3]
    # T = data[0:data_num,0]
    for i in range(data_num):
        T[i] = data[i][0]
        gpsmat[i][0] = data[i][1]
        gpsmat[i][1] = data[i][2]
        gpsmat[i][2] = data[i][3]
    # print(gpsmat)
    # print(T)

    return T, data, gpsmat


# get time check
def read_time(filename="./data2/DX_ZY3_NAD_imagingTime.txt"):
    f = open(filename, 'r')
    data = []
    f.readline()
    for lines in f:
        x = float(lines.split('\t')[1])
        # y = float(lines.split('\t')[2])

        data.append(x)
    # print(data[0])
    f.close()
    return data


# get look direction
def read_cbr(filename="./data2/NAD.cbr"):
    f = open(filename, 'r')
    data = []

    f.readline()
    for lines in f:
        x = float(lines.split('\t')[1])
        y = float(lines.split('\t')[2])

        data.append([x, y, 1])

    f.close()
    # print(data,data.__len__())
    return data


# get rotation matrix from J2000 to WGS84
def read_J2W(filename="./data2/J2WGS.txt"):
    f = open(filename, 'r')
    time = []
    R = []
    rmat = np.zeros((12, 3))
    T = np.zeros(4)

    for lines in f:
        timeCode = float(lines.split(' ')[2])
        r = np.zeros((3, 3))
        line = f.readline()
        r[0][0] = float(line.split('\t')[0])
        r[0][1] = float(line.split('\t')[1])
        r[0][2] = float(line.split('\t')[2])
        line = f.readline()
        r[1][0] = float(line.split('\t')[0])
        r[1][1] = float(line.split('\t')[1])
        r[1][2] = float(line.split('\t')[2])
        line = f.readline()
        r[2][0] = float(line.split('\t')[0])
        r[2][1] = float(line.split('\t')[1])
        r[2][2] = float(line.split('\t')[2])

        time.append(timeCode)
        R.append(r)

    for i in range(4):
        T[i] = time[i]
        rmat[i*3:i*3+3, 0:3] = R[i]

    f.close()

    # print(rmat)

    return T, R, rmat


# 读取相机卫星姿态关系
def read_nad(filename="./data2/NAD.txt"):
    f = open(filename)
    data = np.zeros(6)
    f.readline()
    data[0] = f.readline().replace('\n', '').split(' ')[2]
    data[1] = f.readline().replace('\n', '').split(' ')[2]
    data[2] = f.readline().replace('\n', '').split(' ')[2]
    data[3] = f.readline().replace('\n', '').split(' ')[2]
    data[4] = f.readline().replace('\n', '').split(' ')[2]
    data[5] = f.readline().replace('\n', '').split(' ')[2]
    return data


if __name__ == "__main__":
    # get_XYZ()
    # read_J2W()
    # read_gps()
    # read_time()
    # read_att()
    # read_cbr()
    pass
