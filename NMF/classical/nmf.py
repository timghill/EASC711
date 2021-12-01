import numpy as np
from matplotlib import pyplot as plt

img = plt.imread('./data/small.jpg')
V = img[:, :, 1]


def euclidean(M):
    return np.sum(M**2)

def divergence(A, B):
    pass

class NMF:
    def __init__(self, V, r):
        self.V = V
        self.r = r

        # INITIALIZATION
        np.random.seed(50)
        n, m = V.shape
        self.W = 0.5 + 0.5*np.random.random((n, r))
        self.H = 0.5 + 0.5*np.random.random((r, m))

    def update_H(self):
        H = self.H
        W = self.W
        V = self.V
        H = H*(np.matmul(W.T, V)/np.matmul(W.T, np.matmul(W, H)))
        self.H = H

    def update_W(self):
        H = self.H
        W = self.W
        V = self.V
        W = W*(np.matmul(V, H.T))/np.matmul(W, np.matmul(H, H.T))
        self.W = W

    def residual(self):
        return euclidean(self.V - np.matmul(self.W, self.H))

    def solve(self, rtol=1e-4):
        self.update_W()
        self.update_H()

        resid = self.residual()
        rerr = 1
        residuals = [resid]

        while rerr>=rtol:
        # for i in range(250):
            self.update_W()
            self.update_H()

            new_resid = self.residual()
            rerr = (resid - new_resid)/residuals[0]
            resid = new_resid
            residuals.append(new_resid)

        return self.W, self.H, residuals

distance_fig, distance_ax = plt.subplots()

sample_rank_images = [0, 0, 0]

for r in range(5, 21):
    print(r)
    nmf = NMF(V, r)
    W, H, distance = nmf.solve()

    distance_ax.plot(distance, label='r = %d' % r)

    if r==5:
        sample_rank_images[0] = np.matmul(W, H)

    elif r==15:
        sample_rank_images[1] = np.matmul(W, H)

    elif r==20:
        sample_rank_images[2] = np.matmul(W, H)

distance_ax.legend()
distance_ax.set_xlabel('Iterations')
distance_ax.set_ylabel('Least squares error')
plt.tight_layout()
distance_fig.savefig('nmf_errors.png', dpi=600)

# print(sample_rank_images[2].shape)
fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(8, 8))
axs = axs.flatten()
rvals = [5, 15, 20]
for k in range(3):
    axs[k].imshow(sample_rank_images[k], cmap='gray')
    axs[k].set_title('r = %d' % rvals[k])

axs[-1].imshow(V, cmap='gray')
axs[-1].set_title('Original')
plt.tight_layout()

fig.savefig('nmf_images.png', dpi=600)
plt.show()
