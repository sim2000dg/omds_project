from abc import abstractmethod
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix, solvers
from typing import Optional, Dict
import time


class GaussianSVM:
    """
    Class implementing a Gaussian Kernel SVM, the available fit uses cvxopt and the optimization of the dual problem.
    """

    def __init__(self, inv_reg, gamma) -> None:
        """
        Initialization of a Gaussian kernel SVM object. `inv_reg`  is the inverse regularization coefficient and `gamma`
        is an hyperparameter of the gaussian kernel inducing the implicit mapping of the feature vectors to the unit
        sphere of the l_2 sequence space.

        Higher values of gamma drive the resulting embeddings towards orthogonality between one another (trivial and
        meaningless linear separability), low values of gamma drive the embeddings towards linear dependence
        (and thus non separability, at least numerically).
        :param inv_reg: The inverse regularization, a linear scaling of the hinge loss in the primal objective.
        :param gamma: Hyperparameter of the RBF/Gaussian kernel. An improper value can create numerical problems.
        """
        self.support: Optional[np.ndarray] = None
        self.support_labels: Optional[np.ndarray] = None
        self.support_dual_vars: Optional[np.ndarray] = None
        self.inv_reg: float = float(inv_reg)
        self.gamma: float = gamma
        self.intercept: Optional[float] = None

    def cvxopt_fit(self, train_data: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Fitting of the RBF SVM by solving the dual problem with CVXOPT quadratic programming utilities.
        Relevant dual variables/primal multipliers, labels and observations are saved in order to allow prediction.
        :param train_data: Array of train data, with shape (`n_obs`, `n_features`).
        :param labels: The array of response labels, encoded as \{-1, 1\}.
        :return: The dictionary with information on the optimization process.
        """

        if not np.all(np.sort(np.unique(labels)) == [-1, 1]):
            raise ValueError("The labels need to have a \{-1, 1\} encoding.")
        gram = matrix(
            rbf_kernel(train_data, gamma=self.gamma) * np.outer(labels, labels),
            tc="d",
        )
        A = matrix(labels[np.newaxis, :])  # Equality constraints matrix, a row vector
        q = matrix(
            -np.ones(len(labels), dtype=np.float64)
        )  # Linear term of the objective function
        b = matrix([0], tc="d")  # Equality constraint value, a single zero
        h = matrix(
            np.concatenate(
                [
                    np.zeros(len(labels), dtype=np.float64),
                    np.repeat(self.inv_reg, len(labels)),
                ]
            ),
            tc="d",
        )  # The vector for the values for the inequality constraints, a sequence of zeros and inverse reg. coefs
        # The sparse matrix for the inequality constraints, a block matrix composed of two diagonal sub-matrices of 1s
        # and -1s disposed vertically
        G = spmatrix(
            [-1] * len(labels) + [1] * len(labels),
            I=range(2 * len(labels)),
            J=2 * list(range(len(labels))),
            tc="d",
        )
        init_vars = {
            "x": matrix(np.zeros(len(labels)), tc="d")
        }  # Dict for setting the initial values for the solver

        solvers.options["show_progress"] = False
        start = time.process_time()  # Timer start
        dual_sol = qp(gram, q, G, h, A, b, initvals=init_vars)
        elapsed = time.process_time() - start  # Timer end

        dual_vars = np.squeeze(np.array(dual_sol["x"]))  # Get solution of the dual
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

        # Compute KKT violation in the usual way, we need to add some tolerance here since the solver does not set
        # precisely to the boundary the dual variables; an absolute tolerance of 10^-6*C is chosen to tackle this.
        check_viol = -((gram @ dual_vars) - 1) / labels
        mask_zero = ~support_mask
        mask_reg = np.isclose(dual_vars, self.inv_reg, atol=1e-6, rtol=1e-10)
        mask_r = (mask_zero & (labels == 1)) | (mask_reg & (labels == -1))
        mask_s = (mask_zero & (labels == -1)) | (mask_reg & (labels == +1))
        r_set = mid_mask | mask_r
        s_set = mid_mask | mask_s
        violation = np.max(check_viol[r_set]) - np.min(check_viol[s_set])

        opt_dict = {
            "InitObj": float(0),
            "FinalOpt": dual_sol["dual objective"],
            "Iterations": dual_sol["iterations"],
            "Time": round(elapsed, 5),
            "KKTViolation": violation,
        }  # Return dictionary
        return opt_dict

    def predict(self, data: np.ndarray, score: bool = False) -> np.ndarray:
        """
        Prediction method for the fitted object. Raises an error if called without fitting (obviously).
        :param data: The data to predict on.
        :param score: Whether to return the decision function instead of its sign. Defaults to False.
        :return: The array of predictions.
        """
        if self.support is None:
            raise AttributeError(
                "The necessary attributes for predicting are not initialized, this means \n"
                "that you have not fitted the SVM model first. Fit the model, then try again."
            )
        inner_prods = rbf_kernel(self.support, data, gamma=self.gamma)
        preds = (
            np.sum(
                inner_prods
                * self.support_labels[:, np.newaxis]
                * self.support_dual_vars[:, np.newaxis],
                axis=0,
            )
            + self.intercept
        )
        preds = np.sign(preds) if not score else preds
        return preds

    @abstractmethod
    def smo_fit(
        self, train_data: np.ndarray, labels: np.ndarray, tol: float, max_iter: int
    ) -> Dict:
        ...
