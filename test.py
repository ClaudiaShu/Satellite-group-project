import project
import RPC
import numpy as np

data = project.virt_grid(25, 25, 30).T
# print(data[0,:])


'''
X = np.array([-2282126.083, -2214395.2, -2284383.02])
Y = np.array([5054348.290, 5882759.5, 5059346.85])
Z = np.array([3142026.214, 2780867.2, 3132391.39])
r = np.array([5820, 5670, 5500])
c = np.array([25, 20, 10])
'''
X = data[0, :]
Y = data[1, :]
Z = data[2, :]
r = data[3, :]
c = data[4, :]

J = RPC.calRPC(X, Y, Z, r, c)
'''
for i in range(78):
    print(J[i])
'''
print(J)
