import numpy as np
import numpy.typing as npt
import numpy.linalg as la
from scipy.linalg import eigh

class MyCSP():
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components # the number of components to decompose EEG signals
        self.reg = reg                   #
        self.log = log                   #
        self.norm_trace = norm_trace     # normalize class covariance by its trace

    # X: (n_epochs, n_channels, n_times)
    def class_cov(self, X: npt.NDArray[np.float64]):
        n_channels = X.shape[1]
        Z = X.transpose(1, 0, 2).reshape(n_channels, -1)
        return np.matmul(Z, Z.T) / (Z.shape[1] - 1)

    # X: (n_epochs, n_channels, n_times)
    # y: (n_epochs,)
    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.int_]):
        S_p = self.class_cov(X[y == 0])
        S_n = self.class_cov(X[y == 1])

        evals, evecs = eigh(S_p, S_p + S_n)

        order = np.argsort(evals)[::-1]
        self.evals_ = evals[order]
        self.filters_ = evecs[:, order].T
        return self

    # X: (n_epochs, n_channels, n_times)
    def transform(self, X: npt.NDArray[np.float64]):
        m = self.n_components // 2
        W_selected = np.concatenate([self.filters_[:m, :], self.filters_[-m:, :]], axis=0)

        X_prj = np.matmul(W_selected, X)
        features = (X_prj**2).mean(axis=2)

        if self.log:
            features = np.log(features)

        return features
