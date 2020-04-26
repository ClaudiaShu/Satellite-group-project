read_att(filename="./data/DX_ZY3_NAD_att.txt", flag=0)
用于读取att文件
flag默认值为0，用于读取资源三数据，返回[(timeCode, q1, q2, q3),...]
flag设为1时，用于读取仿真数据，返回[(timeCode, q1, q2, q3, q4),...]

read_gps(filename="./data/DX_ZY3_NAD_gps.txt")
用于读取gps数据
返回[(timCode, PX, PY, PZ, VX, VY, VZ),...]

read_time(filename="./data/DX_ZY3_NAD_imagingTime.txt", flag=0)
用于读取时间数据文件
flag默认值为0，用于读取资源三数据
flag设为1时，用于读取仿真数据
返回[(t, dt),...]

read_cbr(filename="./data/NAD.cbr", flag=0)
用于读取视向量文件
flag默认值为0，用于读取资源三数据
flag设为1时，用于读取仿真数据
返回值为[(φy, φx),...]

read_nad(filename="./data/NAD.txt")
用于读取相机相对卫星本体的姿态参数
返回值为[(pitch, Vpitch, roll, Vroll, yaw, Vyaw),...]