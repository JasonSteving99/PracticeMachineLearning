'''
========================
3D surface (solid color)
========================

Demonstrates a very basic plot of a 3D surface using a solid color.
'''

from math import log

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def H(X, b = 2):
    set_entropy = lambda p : 0 if p == 0 else p * log(1./p, b)
    return sum(map(set_entropy, X/sum(X)))


fig = plt.figure()
ax = fig.gca(projection='3d')

tri_value_points = np.array([np.array([x,y,20 - x - y]) for x in range(21) for y in range(21) if x + y <= 20])
binary_value_points = np.array([np.array([x, 20 - x]) for x in range(21)])

tri_value_entropy_distribution = list(map(lambda xs : H(xs, b = 3), tri_value_points))
ax.plot_trisurf(tri_value_points[:,0]/20, tri_value_points[:,1]/20, tri_value_entropy_distribution, color='b')
tri_value_avg_entropy = sum(tri_value_entropy_distribution)/len(tri_value_points)
ax.plot(tri_value_points[:,0]/20, [0]*len(tri_value_points), tri_value_avg_entropy, color='black', ls=':')
ax.plot(binary_value_points[:,0]/20, [0]*len(binary_value_points), list(map(H, binary_value_points)), color='r')
ax.plot(binary_value_points[:,0]/20, [0]*len(binary_value_points), sum(list(map(H, binary_value_points)))/len(binary_value_points), color='black', ls='--')
ax.plot(tri_value_points[:,0]/20, [0]*len(tri_value_points), tri_value_avg_entropy/log(2,3), color='g', ls=':')

ax.set_xlabel('P(A=x)')
ax.set_ylabel('P(A=y)')
ax.set_zlabel('H(A)')
plt.show()
