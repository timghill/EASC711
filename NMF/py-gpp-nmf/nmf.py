import numpy as np

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
