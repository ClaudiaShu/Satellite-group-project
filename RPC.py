# -*- coding: utf-8 -*-
"""
Created on Wed May 13 16:55:14 2020

@author: Moly
"""

import numpy as np
from project import virt_grid
from scipy import linalg
data = virt_grid(11,11,3)


X=data[:,1]
Y=data[:,1]
Z=data[:,1]
r=data[:,1]
c=data[:,1]
#正投影之后的输入

m=X.shape[0]

#归一化
r0=np.sum(r)/m
c0=np.sum(c)/m
X0=np.sum(X)/m
Y0=np.sum(Y)/m
Z0=np.sum(Z)/m


rs=max(np.max(r)-r0,-np.min(r)+r0)
cs=max(np.max(c)-c0,-np.min(c)+c0)
Xs=max(np.max(X)-X0,-np.min(X)+X0)
Ys=max(np.max(Y)-Y0,-np.min(Y)+Y0)
Zs=max(np.max(Z)-Z0,-np.min(Z)+Z0)


rn=(r-r0)/rs
cn=(c-c0)/cs
Xn=(X-X0)/Xs
Yn=(Y-Y0)/Ys
Zn=(Z-Z0)/Zs

#行列同时解求
p=np.array([np.ones(m),Zn,Yn,Xn,Zn*Yn,Zn*Xn,Yn*Xn,Zn*Zn,Yn*Yn,Xn*Xn,Zn*Yn*Xn,Zn*Zn*Yn,Zn*Zn*Xn,Yn*Yn*Zn,Yn*Yn*Xn,Zn*Xn*Xn,Yn*Xn*Xn,Zn*Zn*Zn,Yn*Yn*Yn,Xn*Xn*Xn]).transpose()

Mr=np.hstack((p,np.repeat([-rn],19,axis=0).transpose()*p[:,1:20]))
Mc=np.hstack((p,np.repeat([-cn],19,axis=0).transpose()*p[:,1:20]))

Z = np.zeros((m,39)) 

M = np.asarray(np.bmat([[Mr, Z], [Z, Mc]]))
R=np.hstack((rn,cn))
W=np.eye(m+m)#设置单位矩阵为初值


i=1
v=5*np.ones([m*2,])
while np.max(abs(v))>1e-5 and i<300:
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
    c=J[39:59]
    d=np.hstack(([1],J[59:]))
    
    B=1.0/(np.dot(p,b)+1e-5)
    D=1.0/(np.dot(p,d)+1e-5)
    W=np.diag(np.hstack((B,D)))#更新参数
    
    i+=1
    print(i)
    
print(J)#J是解求的系数





