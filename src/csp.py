import numpy as np
import numpy.typing as npt
import numpy.linalg as la
from mne.decoding import CSP

# MyCSP = CSP

class MyCSP():
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components # the number of components to decompose EEG signals
        self.reg = reg                   #
        self.log = log                   #
        self.norm_trace = norm_trace      # normalize class covariance by its trace.

    def mean_cov(self, X):
        _, _, times = X.shape
        covs_per_epoch = np.matmul(X, np.transpose(X, (0, 2, 1))) / (times - 1)
        return np.mean(covs_per_epoch, axis=0)

    def fit(self, X: npt.NDArray[np.int_], y: npt.NDArray[np.int_]):
        X_p = X[y == 0]
        X_n = X[y == 1]
        S_p = self.mean_cov(X_p)
        S_n = self.mean_cov(X_n)

        M = la.inv(S_n) @ S_p
        [L, W] = la.eig(M)

        sorted_indices = np.argsort(L.real)[::-1]
        W_sorted = W[:, sorted_indices]
        self.filters_ = W_sorted

        return self

    def transform(self, X):
        W_T = np.transpose(self.filters_)
        m = self.n_components // 2
        W_selected = np.concatenate([W_T[:m, :], W_T[-m:, :]], axis=0)

        X_prj = W_selected @ X

        features = np.var(X_prj, axis=2)

        if self.log:
            features = np.log(features)

        return features
