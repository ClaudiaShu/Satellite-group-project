# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:55:14 2020

@author: Moly
"""

import numpy as np
from project import virt_grid
import pandas as pd

#归一化hanshu
def standard(data):
    m=data.shape[0]
    #归一化
    data0=np.sum(data)/m
    datas=max(np.max(data)-data0,-np.min(data)+data0)
    datan=(data-data0)/datas
    return datan

def RPC(data):
    
    X=data[:,1]
    Y=data[:,1]
    Z=data[:,1]
    r=data[:,1]
    c=data[:,1]
    #正投影之后的输入
    m=X.shape[0]
    
    rn=standard(r)
    cn=standard(c)
    Xn=standard(X)
    Yn=standard(Y)
    Zn=standard(Z)
    #行列同时解求
    p=np.array([np.ones(m),Zn,Yn,Xn,Zn*Yn,Zn*Xn,Yn*Xn,Zn*Zn,Yn*Yn,Xn*Xn,Zn*Yn*Xn,Zn*Zn*Yn,Zn*Zn*Xn,Yn*Yn*Zn,Yn*Yn*Xn,Zn*Xn*Xn,Yn*Xn*Xn,Zn*Zn*Zn,Yn*Yn*Yn,Xn*Xn*Xn]).transpose()

    Mr=np.hstack((p,np.repeat([-rn],19,axis=0).transpose()*p[:,1:20]))
    Mc=np.hstack((p,np.repeat([-cn],19,axis=0).transpose()*p[:,1:20]))
    
    Z = np.zeros((m,39)) 
    
    M = np.asarray(np.bmat([[Mr, Z], [Z, Mc]]))
    R=np.hstack((rn,cn))
    W=np.eye(m+m)#设置单位矩阵为初值
    

    i=0
    v=5*np.ones([m*2,])
    while np.max(abs(v))>1e-15 and i<10:
        s1=np.dot(M.transpose(),W)
        s2=np.dot(s1,W)
        left=np.dot(s2,M)
        right=np.dot(s2,R)
        #s3=np.linalg.det(left)
        #print(left)
        J=np.dot(np.linalg.pinv(left),right)
        #J = linalg.solve(left,right)
        v=np.dot(np.dot(W,M),J)-np.dot(W,R)
        #print(v)
        print(np.max(abs(v)))
        a=J[:20]
        b=np.hstack(([1],J[20:39]))
        C=J[39:59]
        d=np.hstack(([1],J[59:]))
        
        B=1.0/(np.dot(p,b)+1e-5)
        D=1.0/(np.dot(p,d)+1e-5)
        W=np.diag(np.hstack((B,D)))#更新参数
        
        i+=1
        print(i)
        
    print(J)#J是解求的系数a,b,c,d
    return a,b,C,d

def calrc(a,b,c,d,p):
    Rn=np.dot(a,p)/np.dot(b,p)
    Cn=np.dot(C,p)/np.dot(d,p)#计算坐标
    return Rn,Cn

if __name__ == "__main__":
    data = virt_grid(11,11,3)
    a,b,C,d=RPC(data)
    valid=pd.read_csv("bk_projection.csv")
    X=np.array(valid['X'])
    Y=np.array(valid['Y'])
    Z=np.array(valid['Z'])
    r=np.array(valid['x'])
    c=np.array(valid['y'])
    m=X.shape[0]
    rn=standard(r)
    cn=standard(c)#这两个是原坐标归一化后的结果
    Xn=standard(X)
    Yn=standard(Y)
    Zn=standard(Z)
    p=np.array([np.ones(m),Zn,Yn,Xn,Zn*Yn,Zn*Xn,Yn*Xn,Zn*Zn,Yn*Yn,Xn*Xn,Zn*Yn*Xn,Zn*Zn*Yn,Zn*Zn*Xn,Yn*Yn*Zn,Yn*Yn*Xn,Zn*Xn*Xn,Yn*Xn*Xn,Zn*Zn*Zn,Yn*Yn*Yn,Xn*Xn*Xn])
    Rn,Cn=calrc(a,b,C,d,p)#算出来的归一化坐标
    
    print(Rn[-5:])
    print(rn[-5:])
    print(Cn[-5:])
    print(cn[-5:])
