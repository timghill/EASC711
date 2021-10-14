"""
Testing pyDOE module latin hypercube sampling
"""

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
import pyDOE

X = pyDOE.lhs(2, samples=21, criterion='maximin', iterations=1)
Y = pyDOE.lhs(2, samples=21, criterion='cm', iterations=100)

fig, ax = plt.subplots()
ax.plot(X[:,0], X[:,1], '.')
ax.plot(Y[:, 0], Y[:,1], '.')

def pairwise_dist(Z, unique=True):
    N = Z.shape[0]
    N_unique = int(0.5*N*(N-1))

    if unique is False:
        D = np.zeros((N, N))
    D_unique = np.zeros(N_unique)
    ind_unique = 0
    for i in range(N):
        for j in range(i+1,N):
            dist_ij = np.linalg.norm(Z[i,:] - Z[j,:])
            D_unique[ind_unique] = dist_ij
            if unique is False:
                D[i,j] = dist_ij
            ind_unique += 1
    if unique is False:
        return D
    else:
        return D_unique

def max_min_dist(Z):
    D = pairwise_dist(Z, unique=False)[:-1]
    D[D==0] = np.nan
    D_pairwise_min = np.nanmin(D, axis=1)
    D_maximin = np.nanmin(D_pairwise_min)
    return D_maximin


# print(pairwise_dist(X))
print(max_min_dist(X))
print(max_min_dist(Y))

plt.show()
