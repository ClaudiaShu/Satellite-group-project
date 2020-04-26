read_att(filename="./data/DX_ZY3_NAD_att.txt", flag=0)
用于读取att文件
flag默认值为0，用于读取资源三数据，返回[(timeCode, q1, q2, q3),...]
flag设为1时，用于读取仿真数据，返回[(timeCode, q1, q2, q3, q4),...]

read_gps(filename="./data/DX_ZY3_NAD_gps.txt")
用于读取gps数据
返回[(timCode, PX, PY, PZ, VX, VY, VZ),...]（array）

read_time(filename="./data/DX_ZY3_NAD_imagingTime.txt", flag=0)
用于读取时间数据文件
flag默认值为0，用于读取资源三数据
flag设为1时，用于读取仿真数据
返回[(t, dt),...]（array）

read_cbr(filename="./data/NAD.cbr", flag=0)
用于读取视向量文件
flag默认值为0，用于读取资源三数据
flag设为1时，用于读取仿真数据
返回值为[(φy, φx),...]（list）

read_nad(filename="./data/NAD.txt")
用于读取相机相对卫星本体的姿态参数
返回值为[(pitch, Vpitch, roll, Vroll, yaw, Vyaw),...](list)

rotation_matrix(pitch, rol, yaw)
用于计算旋转矩阵
返回旋转矩阵（array）

q_interpolation(att1, att2, t, flag=0)
用于进行四元数插值
att1 att2分别为读取数据文件得到的包含时间信息的数据条
flag默认为0 读取仿真文件设置为1
返回四元数(q1,q2,q3,q4)(array)