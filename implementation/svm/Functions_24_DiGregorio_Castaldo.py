from .Functions_23_DiGregorio_Castaldo import GaussianSVM
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import time


class GaussianSVMComplete(GaussianSVM):
    def smo_fit(
        self, train_data: np.ndarray, labels: np.ndarray, tol: float, max_iter: int
    ):
        """
        Fitting of the RBF SVM by solving the dual problem with a SMO decomposition algorithm
        with maximum violating pairs chosen at each iteration.
        Relevant dual variables/primal multipliers, labels and observations are saved in order to allow prediction.
        :param train_data: Array of train data, with shape (`n_obs`, `n_features`).
        :param labels: The array of response labels, encoded as \{-1, 1\}.
        :param tol: The tolerance for the KKT violation for maximum violating pair SMO.
        :param max_iter: The maximum number of iterations.
        :return: The dictionary with information on the optimization process.
        """
        start = time.process_time()  # Timer start
        if not np.all(np.sort(np.unique(labels)) == [-1, 1]):
            raise ValueError("The labels need to have a \{-1, 1\} encoding.")

        current_gradient = np.full(
            fill_value=-1, shape=len(train_data), dtype=np.float64
        )  # Initialize current gradient
        current_check = np.zeros_like(
            current_gradient
        )  # Initialize array we are taking max and min over
        dual_vars = np.zeros_like(current_gradient)  # Dual variables array
        grad_update = np.zeros_like(
            current_gradient
        )  # allocate array for gradient update computation
        set_r = labels == 1  # initialize mask for set R
        set_s = labels == -1  # initialize mask for set S
        gram = rbf_kernel(
            train_data, gamma=self.gamma
        )  # compute Gram matrix of inner products
        gram *= np.outer(labels, labels)  # Multiply each entry by y^iy^j
        it = 0  # Iteration counter

        while True:
            if (
                np.sum(set_r) * np.sum(set_s) == 0
            ):  # Check whether one of the two sets is empty, in that case break
                break
            np.multiply(
                current_gradient, labels, out=current_check
            )  # Compute -nabla_alpha(alpha^k)/y
            np.multiply(current_check, -1, out=current_check)

            r_pick = np.where(set_r & (current_check == np.max(current_check[set_r])))[
                0
            ]  # Get the argmax set for R
            s_pick = np.where(set_s & (current_check == np.min(current_check[set_s])))[
                0
            ]  # Get the argmin set for S

            r_pick = r_pick[0]  # pick first element of argmax set for R
            s_pick = s_pick[0]  # same for S

            if (
                current_check[r_pick] - current_check[s_pick] < tol
            ):  # Condition for termination, KKT violation
                break

            # Save previous values for gradient update
            pred_r = dual_vars[r_pick]
            pred_s = dual_vars[s_pick]

            # alpha_s*y^s + alpha_r^y^r = sum_pair for the equality constraint of the problem
            sum_pair = np.dot(dual_vars[[r_pick, s_pick]], labels[[r_pick, s_pick]])
            rel_gram = gram[
                np.ix_([r_pick, s_pick], [r_pick, s_pick])
            ]  # Entries of the hessian related to sel. vars
            # Compute quadratic term of the univariate quadratic loss in alpha_r
            quad_term = (
                (1 / 2) * rel_gram[0, 0]
                - rel_gram[0, 1] * labels[r_pick] * labels[s_pick]
                + (1 / 2) * rel_gram[1, 1]
            )
            # Compute linear term of the univariate quadratic loss in alpha_r
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

            if (
                labels[r_pick] * labels[s_pick] > 0
            ):  # Check if labels are the same, then compute admissible interval
                lower = dual_vars[r_pick] + dual_vars[s_pick] - self.inv_reg
                upper = dual_vars[r_pick] + dual_vars[s_pick]
            else:
                lower = dual_vars[r_pick] - dual_vars[s_pick]
                upper = self.inv_reg - dual_vars[s_pick] + dual_vars[r_pick]

            admissible = np.array([max(0, lower), min(self.inv_reg, upper)])

            # vertex of the parabola
            vertex = -lin_term / (2 * quad_term)
            if quad_term > 0:   # check whether parabola is convex
                # If vertex is inside the admissible interval, vertex is the solution for the iteration
                # On the contrary, if outside, get closer boundary point
                if vertex < admissible[0]:
                    dual_vars[r_pick] = admissible[0]
                elif vertex > admissible[1]:
                    dual_vars[r_pick] = admissible[1]
                else:
                    dual_vars[r_pick] = vertex
            else:     # parabola is concave, get the farthest of the boundary points of the admissible interval
                dual_vars[r_pick] = admissible[np.argmax(np.abs(vertex-admissible))]

            # Compute other variable after having solved for the first
            dual_vars[s_pick] = labels[s_pick] * (
                -dual_vars[r_pick] * labels[r_pick] + sum_pair
            )

            for var in [s_pick, r_pick]:  # Update sets accordingly
                if (dual_vars[var] > 0) and (dual_vars[var] < self.inv_reg):
                    set_r[var] = True
                    set_s[var] = True
                elif ((dual_vars[var] == 0) and (labels[var] > 0)) or (
                    (dual_vars[var] == self.inv_reg) and (labels[var] < 0)
                ):
                    set_r[var] = True
                    set_s[var] = False
                else:
                    set_r[var] = False
                    set_s[var] = True

            # Update gradient
            np.multiply(gram[:, r_pick], dual_vars[r_pick] - pred_r, out=grad_update)
            np.add(current_gradient, grad_update, out=current_gradient)
            np.multiply(gram[:, s_pick], dual_vars[s_pick] - pred_s, out=grad_update)
            np.add(current_gradient, grad_update, out=current_gradient)

            it += 1
            if it == max_iter:  # Check maximum number of iterations
                break

        elapsed = time.process_time() - start
        # Mask for support
        support_mask = ~np.isclose(
            dual_vars, 0, rtol=1e-10, atol=1e-6
        )  # Mask for support vectors
        mid_mask = support_mask & (
            ~np.isclose(dual_vars, self.inv_reg, rtol=1e-10, atol=1e-6)
        )  # Mask for support vectors that we are sure are not misclassified and are useful to get b out
        self.support = train_data[support_mask]  # Save support vector matrix
        self.support_labels = labels[support_mask]  # Save support vector labels
        self.support_dual_vars = dual_vars[support_mask]  # Save relevant dual variables
        self.intercept = np.mean(
            labels[mid_mask]
            - np.dot(
                (gram * np.outer(labels, labels))[np.ix_(mid_mask, support_mask)],
                labels[support_mask] * dual_vars[support_mask],
            ),
            axis=0,
        )  # Save intercept

        final_obj = (1 / 2) * (dual_vars @ gram @ dual_vars) - np.sum(
            dual_vars
        )  # Compute final objective function
        opt_dict = {
            "InitObj": 0,
            "FinalOpt": final_obj,
            "Iterations": it,
            "Time": elapsed,
            "KKTViolation": 0
            if (np.sum(set_r) * np.sum(set_s) == 0)
            else current_check[r_pick] - current_check[s_pick],
        }
        return opt_dict
