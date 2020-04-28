import numpy as np
import cv2
# from libtiff import TIFF




#get att check
def read_att(filename="./data/DX_ZY3_NAD_att.txt"):
    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(6):
            f.readline()
        data_num = f.readline().split(' ')[2]
        data_num = int(data_num)#401

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
        return data

#get gps check
def read_gps(filename="./data/DX_ZY3_NAD_gps.txt"):
    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(4):
            f.readline()
        data_num = f.readline().split(' ')[2]#101
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
            data.append([timeCode, PX, PY, PZ, VX, VY, VZ])

            f.readline()
        return data

#get time check
def read_time(filename="./data/DX_ZY3_NAD_imagingTime.txt"):
    f = open(filename, 'r')
    data = []

    f.readline() 
    for lines in f:
        x = float(lines.split('\t')[1])
        y = float(lines.split('\t')[2])

        data.append([x, y])

    f.close()
    return data

#get look direction
def read_cbr(filename="./data/NAD.cbr"):
    f = open(filename, 'r')
    data = []

    f.readline() 
    for lines in f:
        x = float(lines.split('\t')[1])
        y = float(lines.split('\t')[2])

        data.append([x, y])

    f.close()
    return data

#get rotation matrix from J2000 to WGS84
def read_J2W(filename="./data/J2WGS.txt"):
    f = open(filename, 'r')
    time = []
    R = []

    for lines in f:
        timeCode = float(lines.split(' ')[2])
        r = np.zeros((3,3))
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
    T = np.zeros(4)
    for i in range(4):
        T[i] = time[i]

    f.close()
    return T,R

#获得像点坐标 读取DEM tiff文件
def get_XYZ(filename="./data/SRTM_mosaic.tif"):
    # img = cv2.imread(filename)

    #####test data######
    data = []
    data.append([1934303.0,743483.0,5297.0])
    data.append([2343434.0,798337.0,675.0])
    data.append([0.0,0.0,0.0])



    return data

if __name__ == "__main__":
    # get_XYZ()
    # read_J2W()
    pass
