from Functions_23_DiGregorio_Castaldo import GaussianSVM
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np


class GaussianSVMComplete(GaussianSVM):
    def smo_fit(
        self, train_data: np.ndarray, labels: np.ndarray, tol: float, max_iter: int
    ):
        """
        Fitting of the RBF SVM by solving the dual problem with an SMO algorithm with maximum violating pairs chosen
        at each iteration.
        Relevant dual variables/primal multipliers, labels and observations are saved in order to allow prediction.
        :param train_data: Array of train data, with shape (`n_obs`, `n_features`).
        :param labels: The array of response labels, encoded as \{-1, 1\}.
        :param tol: The tolerance for the KKT violation for maximum violating pair SMO.
        :param max_iter: The maximum number of iterations.
        :return: The dictionary with information on the optimization process.
        """
        current_gradient = np.full(
            fill_value=-1, shape=len(train_data), dtype=np.float64
        )
        current_check = np.zeros_like(current_gradient)
        dual_vars = np.zeros_like(current_gradient)
        grad_update_1 = np.zeros_like(current_gradient)
        grad_update_2 = np.zeros_like(current_gradient)
        set_r = labels == 1
        set_s = labels == -1
        gram = rbf_kernel(train_data, gamma=self.gamma)
        gram *= np.outer(labels, labels)
        it = 0

        while True:
            if np.sum(set_r) * np.sum(set_s) == 0:
                break
            np.multiply(current_gradient, labels, out=current_check)
            np.multiply(current_check, -1, out=current_check)

            r_pick = np.where(set_r & (current_check == np.max(current_check[set_r])))[
                0
            ]
            s_pick = np.where(set_s & (current_check == np.min(current_check[set_s])))[
                0
            ]

            r_pick = r_pick[0]
            s_pick = s_pick[0] if s_pick[0] != r_pick else s_pick[1]

            if current_check[r_pick] - current_check[s_pick] < tol:
                break

            pred_r = dual_vars[r_pick]
            pred_s = dual_vars[s_pick]

            sum_pair = np.dot(dual_vars[[r_pick, s_pick]], labels[[r_pick, s_pick]])
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
                * labels[s_pick]
                * (
                    np.dot(gram[s_pick], dual_vars)
                    - gram[s_pick, s_pick] * dual_vars[s_pick]
                )
                + np.dot(gram[r_pick], dual_vars)
                - gram[r_pick, r_pick] * dual_vars[r_pick]
                - gram[r_pick, s_pick]
                * (
                    dual_vars[s_pick]
                    - labels[r_pick] * labels[s_pick] * dual_vars[r_pick]
                )
                - 1
                + labels[r_pick] * labels[s_pick]
            )

            prop = -lin_term / (2 * quad_term)

            if labels[r_pick] * labels[s_pick] > 0:
                lower = dual_vars[r_pick] + dual_vars[s_pick] - self.inv_reg
                upper = dual_vars[r_pick] + dual_vars[s_pick]
            else:
                lower = dual_vars[r_pick] - dual_vars[s_pick]
                upper = self.inv_reg - dual_vars[s_pick] + dual_vars[r_pick]

            admissible = np.array([max(0, lower), min(self.inv_reg, upper)])
            if prop < admissible[0]:
                dual_vars[r_pick] = admissible[0]
            elif prop > admissible[1]:
                dual_vars[r_pick] = admissible[1]
            else:
                dual_vars[r_pick] = prop

            dual_vars[s_pick] = labels[s_pick] * (
                -dual_vars[r_pick] * labels[r_pick] + sum_pair
            )

            for var in [s_pick, r_pick]:
                if (dual_vars[var] > 0) and (dual_vars[var] < self.inv_reg):
                    set_r[var] = True
                    set_s[var] = True
                elif ((dual_vars[var] == 0) and (labels[var] > 0)) or (
                    (dual_vars[var] == self.inv_reg) and (labels[var] < 0)
                ):
                    set_r[var] = True
                    set_s[var] = False
                elif (dual_vars[var] == 0 and (labels[var] < 0)) or (
                    (dual_vars[var] == self.inv_reg) and (labels[var] > 0)
                ):
                    set_r[var] = False
                    set_s[var] = True

            np.multiply(gram[:, r_pick], dual_vars[r_pick] - pred_r, out=grad_update_1)
            np.multiply(gram[:, s_pick], dual_vars[s_pick] - pred_s, out=grad_update_2)
            np.add(current_gradient, grad_update_1, out=current_gradient)
            np.add(current_gradient, grad_update_2, out=current_gradient)

            it += 1
            if it == max_iter:
                break

        support_mask = ~np.isclose(dual_vars, 0, rtol=1e-10, atol=1e-6)
        mid_mask = support_mask & (
            ~np.isclose(dual_vars, self.inv_reg, rtol=1e-10, atol=1e-6)
        )
        self.support = train_data[support_mask]
        self.support_labels = labels[support_mask]
        self.support_dual_vars = dual_vars[support_mask]
        self.intercept = np.mean(
            labels[mid_mask]
            - np.dot(
                (gram * np.outer(labels, labels))[np.ix_(mid_mask, support_mask)],
                labels[support_mask] * dual_vars[support_mask],
            ),
            axis=0,
        )

        final_obj = (1 / 2) * (dual_vars @ gram @ dual_vars) - dual_vars
        opt_dict = {
            "InitObj": 0,
            "FinalOpt": final_obj,
            "Iterations": it,
            "KKTViolation:": 0
            if (np.sum(set_r) * np.sum(set_s) == 0)
            else current_check[r_pick] - current_check[s_pick],
        }
        return opt_dict


if __name__ == "__main__":
    from implementation.data_import import csv_import

    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(["S", "M"], "../../data.txt", dtype=np.float64)
    mask = train_data[:, -1] == 0
    train_data[mask, -1] = -1
    svm = GaussianSVMComplete(gamma=0.5, inv_reg=1)
    dual_sol = svm.smo_fit(train_data[:, :-1], train_data[:, -1], 1e-5, 1e5)
    # dual_sol_cvxopt = svm.cvxopt_fit(train_data[:, :-1], train_data[:, -1])
    print(np.sum(svm.predict(train_data[:, :-1]) == train_data[:, -1]))

