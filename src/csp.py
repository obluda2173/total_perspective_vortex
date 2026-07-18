import numpy as np
import numpy.typing as npt
import numpy.linalg as la

class MyCSP():
    def __init__(self, n_components=4, reg=None, log=True, norm_trace=False):
        self.n_components = n_components # the number of components to decompose EEG signals
        self.reg = reg                   #
        self.log = log                   #
        self.norm_trace = norm_trace     # normalize class covariance by its trace

    def class_cov(self, X):
        """X: (n_epochs, n_channels, n_times) -> (n_channels, n_channels)"""
        n_channels = X.shape[1]
        Z = X.transpose(1, 0, 2).reshape(n_channels, -1)   # (n_ch, n_epochs * n_times)
        return Z @ Z.T / (Z.shape[1] - 1)

    # X: (n_epochs, n_channels, n_times)
    # y: (n_epochs,)
    def fit(self, X: npt.NDArray[np.float_], y: npt.NDArray[np.int_]):
        X_p = X[y == 0]
        X_n = X[y == 1]
        S_p = self.class_cov(X_p)
        S_n = self.class_cov(X_n)

        M = np.matmul(la.inv(S_p + S_n), S_p)
        # M = np.matmul(la.inv(S_n), S_p)
        [L, W] = la.eig(M)

        sorted_indices = np.argsort(L)[::-1]
        W_sorted = W[:, sorted_indices]
        self.filters_ = np.transpose(W_sorted)
        self.evals_ = L[sorted_indices]

        return self

    def transform(self, X: npt.NDArray[np.float_]):
        m = self.n_components // 2
        W_selected = np.concatenate([self.filters_[:m, :], self.filters_[-m:, :]], axis=0)

        X_prj = np.matmul(W_selected, X)

        features = (X_prj**2).mean(axis=2)

        if self.log:
            features = np.log(features)

        return features
