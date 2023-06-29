import numpy as np
from sklearn.metrics import pairwise_distances
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix
from typing import Optional, Self


class GaussianSVM:
    def __init__(self, inv_reg, gamma):
        self.support: Optional[np.ndarray] = None
        self.support_labels: Optional[np.ndarray] = None
        self.support_dual_vars: Optional[np.ndarray] = None
        self.inv_reg: float = float(inv_reg)
        self.gamma: float = gamma

    def cvxopt_fit(self, train_data: np.ndarray, labels: np.ndarray) -> Self:
        gram = matrix(
            pairwise_distances(
                train_data, metric=self.gaussian_kernel, gamma=self.gamma
            ),
            tc="d",
        )
        A = matrix(labels, (1, len(labels)))
        q = matrix(-np.ones(len(labels), dtype=np.float64))
        b = matrix([0], tc="d")
        h = matrix(
            np.concatenate(
                [
                    np.zeros(len(labels), dtype=np.float64),
                    np.repeat(self.inv_reg, len(labels)),
                ]
            ),
            tc="d",
        )
        G = spmatrix(
            [-1] * len(labels) + [1] * len(labels),
            I=range(2 * len(labels)),
            J=2 * list(range(len(labels))),
            tc="d",
        )

        dual_sol = qp(gram, q, G, h, A, b)
        return dual_sol

    def smo_fit(
        self, train_data: np.ndarray, labels: np.ndarray, tol: float, max_iter: int
    ):
        current_gradient = np.full(
            fill_value=-1, shape=len(train_data), dtype=np.float64
        )
        current_check = np.zeros_like(current_gradient)
        dual_vars = np.zeros_like(current_gradient)
        grad_update_1 = np.zeros_like(current_gradient)
        grad_update_2 = np.zeros_like(current_check)
        set_r = np.full(fill_value=True, shape=len(train_data), dtype=bool)
        set_s = np.copy(set_r)
        set_r[labels > 0] = True
        set_s[labels < 0] = True
        gram = pairwise_distances(
            train_data, metric=self.gaussian_kernel, gamma=self.gamma
        )
        it = 0

        while True:
            np.multiply(current_gradient, labels, out=current_check)
            np.multiply(current_check, -1, out=current_check)

            max_set_r = np.where(
                set_r & (current_check == np.max(current_check[set_r]))
            )[0]
            min_set_s = np.where(
                set_s & (current_check == np.min(current_check[set_s]))
            )[0]
            r_pick = max_set_r[0]
            s_pick = min_set_s[0] if min_set_s[0] != r_pick else min_set_s[1]

            if current_check[r_pick] - current_check[s_pick] < tol:
                break

            sum_pair = -(
                np.dot(labels, dual_vars)
                - np.dot(labels[[r_pick, s_pick]], dual_vars[[r_pick, s_pick]])
            )
            rel_gram = gram[np.ix_([r_pick, s_pick], [r_pick, s_pick])]
            quad_term = (
                (1 / 2) * rel_gram[0, 0]
                - rel_gram[0, 1] * labels[r_pick] * labels[s_pick]
                + (1 / 2) * rel_gram[1, 1]
            )
            lin_term = (
                rel_gram[0, 1] * labels[s_pick] * sum_pair
                - rel_gram[1, 1] * labels[r_pick] * sum_pair
                - labels[r_pick]
                * labels[s_pick]*(
                    np.dot(gram[s_pick], dual_vars)
                    - gram[s_pick, s_pick] * dual_vars[s_pick]
                )
                + np.dot(gram[r_pick], dual_vars)
                - gram[r_pick, r_pick] * dual_vars[r_pick]
                - gram[r_pick, s_pick] * (np.sum(dual_vars[[r_pick, s_pick]]))
                - 1
                + labels[r_pick] * labels[s_pick]
            )

            pred_r = dual_vars[r_pick]
            pred_s = dual_vars[s_pick]

            prop = None
            if quad_term > 0:
                prop = -lin_term / (2 * quad_term)
            if prop is None or ((prop < 0) | (prop > self.inv_reg)):
                prop = (
                    0
                    if 0
                    == min(0, (self.inv_reg ** 2) * quad_term + self.inv_reg * lin_term)
                    else self.inv_reg
                )
            dual_vars[r_pick] = prop
            dual_vars[s_pick] = labels[s_pick] * (
                -dual_vars[r_pick] * labels[r_pick] + sum_pair
            )

            for var in [s_pick, r_pick]:
                if (dual_vars[var] > 0) & (dual_vars[var] < self.inv_reg):
                    set_r[var] = True
                    set_s[var] = True
                elif ((dual_vars[var] == 0) & (labels[var] > 0)) | (
                    (dual_vars[var] == self.inv_reg) & (labels[var] < 0)
                ):
                    set_r[var] = True
                    set_s[var] = False
                else:
                    set_r[var] = False
                    set_s[var] = True

            np.multiply(gram[:, r_pick], dual_vars[r_pick] - pred_r, out=grad_update_1)
            np.multiply(gram[:, s_pick], dual_vars[s_pick] - pred_s, out=grad_update_2)
            np.add(current_gradient, grad_update_1, out=current_gradient)
            np.add(current_gradient, grad_update_2, out=current_gradient)

            it += 1
            if it == max_iter:
                break

        return dual_vars

    @staticmethod
    def gaussian_kernel(x, y, gamma):
        return np.exp(-gamma * np.linalg.norm(x - y))


if __name__ == "__main__":
    from implementation.data_import import csv_import

    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(["S", "M"], "../../data.txt", dtype=np.float64)
    mask = train_data[:, -1] == 0
    train_data[mask, -1] = -1
    svm = GaussianSVM(gamma=0.1, inv_reg=0.1)
    dual_sol = svm.smo_fit(train_data[:, :-1], train_data[:, -1], 1e-5, 1e5)
    print(dual_sol)
