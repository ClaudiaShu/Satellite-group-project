import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

HHH = np.zeros(3)
print(HHH[0])

# 二次函数的标准形式
def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c

# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    return func(params, x) - y

# 对参数求解
def slovePara(X, Y):
    p0 = [10, 10, 10]
    Para = leastsq(error, p0, args=(X, Y))
    return Para

# 输出最后的结果
def solution():
    # 待拟合的数据
    X = np.array([1,2,3,4,5,6])
    Y = np.array([9.1,18.3,32,47,69.5,94.8])
    # Para = slovePara(X, Y)
    # a, b, c = Para[0]
    a, b, c = slovePara(X, Y)[0]
    print("a=",a," b=",b," c=",c)
    # print("cost:" + str(Para[1]))
    print("求解的曲线是:")
    print("y="+str(round(a,2))+"x*x+"+str(round(b,2))+"x+"+str(c))

    plt.figure(figsize=(8,6))
    plt.scatter(X, Y, color="green", label="sample data", linewidth=2)

    # 画拟合直线
    x=np.linspace(0,15,100) ##在0-15直接画100个连续点
    y=a*x*x+b*x+c ##函数式
    plt.plot(x,y,color="red",label="solution line",linewidth=2)
    plt.legend() #绘制图例
    plt.show()


solution()
