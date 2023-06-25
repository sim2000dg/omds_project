import numpy as np
from sklearn.metrics import pairwise_distances
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix
from typing import Optional, Self


class GaussianSVM:
    def __init__(self, inv_reg, gamma):
        self.gram: Optional[np.ndarray] = None
        self.dual_vars: Optional[np.ndarray] = None
        self.inv_reg: float = float(inv_reg)
        self.gamma: float = gamma

    def cvxopt_fit(self, train_data: np.ndarray, labels: np.ndarray) -> Self:
        self.gram = matrix(pairwise_distances(train_data, metric=self.gaussian_kernel, gamma=self.gamma), tc='d')
        A = matrix(labels, (1, len(labels)))
        q = matrix(-np.ones(len(labels), dtype=np.float64))
        b = matrix([0], tc='d')
        h = matrix(np.concatenate([np.zeros(len(labels), dtype=np.float64),
                                   np.repeat(self.inv_reg, len(labels))]), tc='d')
        G = spmatrix([-1]*len(labels)+[1]*len(labels),
                     I=range(2*len(labels)),
                     J=2*list(range(len(labels))),
                     tc='d')

        dual_sol = qp(self.gram, q, G, h, A, b)
        pass

    @staticmethod
    def gaussian_kernel(x, y, gamma):
        return np.exp(-gamma*np.linalg.norm(x-y))


if __name__ == '__main__':
    from implementation.data_import import csv_import
    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(['S', 'M'], '../../data.txt', dtype=np.float64)
    svm = GaussianSVM(gamma=0.1, inv_reg=1)
    svm.cvxopt_fit(train_data[:, :-1], train_data[:, -1])


