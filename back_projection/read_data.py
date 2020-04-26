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

#获得像点坐标 读取DEM tiff文件
def get_XYZ(filename="./data/SRTM_mosaic.tif"):
    img = cv2.imread(filename)
    # print(img.shape)
    # print(img.dtype)

    # tif = TIFF.open(filename,mode='r')
    # lons = 100
    # lone = 137
    # lats = 15
    # late = 52
    # lons_grid = int((lons+180.0)/(30.0/3600))
    # lone_grid = int((lone+180.0)/(30.0/3600))
    # lats_grid = int((75.0-lats)/(30.0/3600))
    # late_grid = int((75.0-late)/(30.0/3600))
    # img2 = img[late_grid:lats_grid,lons_grid:lone_grid]
    # cv2.namedWindow('img')
    # cv2.imshow('img',img2)
    # cv2.waitKey(0)


    #####test data######
    data = []
    data.append([234.0,34.0,675.0])
    data.append([345.0,54.0,597.0])

    return data

if __name__ == "__main__":
    # get_XYZ()
    pass