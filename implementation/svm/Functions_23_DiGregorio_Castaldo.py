from abc import abstractmethod

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from cvxopt.solvers import qp
from cvxopt import matrix, spmatrix, solvers
from typing import Optional, Self
import sys
from typing import Dict


class GaussianSVM:
    """
    Class implementing a Gaussian Kernel SVM, the available fit uses cvxopt and the optimization of the dual problem.
    """

    def __init__(self, inv_reg, gamma):
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
        q = matrix(-np.ones(len(labels), dtype=np.float64))  # Linear term of the objective function
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
        init_vars = {"x": matrix(np.zeros(len(labels)), tc="d")}  # Dict for setting the initial values for the solver

        solvers.options['show_progress'] = False
        dual_sol = qp(gram, q, G, h, A, b, initvals=init_vars)


        dual_vars = np.squeeze(np.array(dual_sol["x"]))  # Get solution of the dual
        support_mask = ~np.isclose(dual_vars, 0, rtol=1e-10, atol=1e-6)  # Mask for support vectors
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
        opt_dict = {
            "InitObj": float(0),
            "FinalOpt": dual_sol["dual objective"],
            "Iterations": dual_sol["iterations"],
            "KKTViolation:": 0,
        }  # Return dictionary
        return opt_dict

    def predict(self, data: np.ndarray):
        """
        Prediction method for the fitted object. Raises an error if called without fitting (obviously).
        :param data: The data to predict on.
        :return: The array of predictions.
        """
        if self.support is None:
            raise AttributeError('The necessary attributes for predicting are not initialized, this means \n'
                                 'that you have not fitted the SVM model first. Fit the model, then try again.')
        inner_prods = rbf_kernel(self.support, data, gamma=self.gamma)
        preds = np.sign(np.sum(inner_prods*self.support_labels[:, np.newaxis]*self.support_dual_vars[:, np.newaxis],
                               axis=0))
        return preds

    @abstractmethod
    def smo_fit(
        self, train_data: np.ndarray, labels: np.ndarray, tol: float, max_iter: int
    ):
        ...


if __name__ == '__main__':
    from implementation.data_import import csv_import

    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(["S", "M"], "../../data.txt", dtype=np.float64)
    mask = train_data[:, -1] == 0
    train_data[mask, -1] = -1
    svm = GaussianSVM(gamma=0.5, inv_reg=1)
    # dual_sol = svm.smo_fit(train_data[:, :-1], train_data[:, -1], 1e-5, 1e5)
    dual_sol_cvxopt = svm.cvxopt_fit(train_data[:, :-1], train_data[:, -1])
    svm.predict(train_data[:, :-1])
    print(dual_sol)
