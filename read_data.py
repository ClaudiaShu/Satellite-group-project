

import numpy as np


def read_att(filename="./data/DX_ZY3_NAD_att.txt", flag=0):  # 如果是用于算仿真数据 flag设为1
    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(6):
            f.readline()
        data_num = f.readline().split(' ')[2]
        data_num = int(data_num)
        data = np.zeros((data_num, 5))

        for i in range(data_num):
            for j in range(2):
                f.readline()
            data[i, 0] = (f.readline().lstrip()).split(' ')[2]
            for j in range(7):
                f.readline()
            data[i, 1] = (f.readline().lstrip()).split(' ')[2]
            data[i, 2] = (f.readline().lstrip()).split(' ')[2]
            data[i, 3] = (f.readline().lstrip()).split(' ')[2]
            if flag:
                data[i, 4] = (f.readline().lstrip()).split(' ')[2]
            f.readline()
        return data


def read_gps(filename="./data/DX_ZY3_NAD_gps.txt"):
    with open(filename) as f:  # with语句自动调用close()方法
        for i in range(4):
            f.readline()
        data_num = f.readline().split(' ')[2]
        data_num = int(data_num)
        data = np.zeros((data_num, 7))
        for i in range(data_num):
            for j in range(2):
                f.readline()
            data[i, 0] = (f.readline().lstrip()).split(' ')[2]
            f.readline()
            data[i, 1] = (f.readline().lstrip()).split(' ')[2]
            data[i, 2] = (f.readline().lstrip()).split(' ')[2]
            data[i, 3] = (f.readline().lstrip()).split(' ')[2]
            data[i, 4] = (f.readline().lstrip()).split(' ')[2]
            data[i, 5] = (f.readline().lstrip()).split(' ')[2]
            data[i, 6] = (f.readline().lstrip()).split(' ')[2]
            f.readline()
        return data


def read_time(filename="./data/DX_ZY3_NAD_imagingTime.txt", flag=0):  # 如果是用于算仿真数据 flag设为1
    data = []
    with open(filename) as f:  # with语句自动调用close()方法
        f.readline()
        line = f.readline().replace('\n', '')
        while line:
            if flag == 0:
                t_data = float(line.split('\t')[4])
                dt_data = float((line.split('\t')[7]))
            else:
                t_data = float(line.split('\t')[1].lstrip())
                dt_data = float(line.split('\t')[2].lstrip())
            data.append((t_data, dt_data))
            line = (f.readline()).replace('\n', '')
        return data


def read_cbr(filename="./data/NAD.cbr", flag=0):
    f = open(filename)
    if flag:
        f.readline()
    line = f.readline().replace('\n', '')
    data = []
    while line:
        py = float(line.split('\t')[1])
        px = float(line.split('\t')[2])
        data.append((py, px))
        line = f.readline().replace('\n', '')
    return data


def read_nad(filename="./data/NAD.txt"):
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






