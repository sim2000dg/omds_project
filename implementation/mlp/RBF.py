import numpy
import numpy as np
from typing import Self, Callable, Tuple

# from ..data_import import csv_import
import math
from typing import Optional, Iterator

from numpy import ndarray
from scipy.optimize import minimize, OptimizeResult
import math
from itertools import product, chain
from time import perf_counter
import pandas as pd
from sklearn.metrics import pairwise_distances


class RBF:

    def __init__(self, train_data: np.ndarray, units: int = 20, rho1: int = 0,
                 rho2: int = 0, sigma: float = 1):
        """
        :param train_data: The training data, as a NumPy array.
        :param units: The number of centers initialized in the hidden layer, as 2-D NumPy array
        :param rho1: The L2 regularization hyperparameter for weights vector. Higher rho, linearly higher L2 penalty.
        :param rho2: The L2 regularization hyperparameter for centers matrix. Higher rho, linearly higher L2 penalty.
        :param sigma: The parameter in the RBF activation function, Multiquadric RBF function is considered
        """
        self.x = train_data
        self.rho1 = rho1
        self.rho2 = rho2
        self.sigma = sigma
        self.units = units
        self.weights = np.random.normal(0, 0.005, size=(units, 1))
        # the centers are randomly picked among the training points
        self.centroids = self.x[np.random.choice(self.x.shape[0], units, replace=False)]

        # array to store upstream gradient in the backpropagation pipeline
        self.gradient = np.zeros(shape=(train_data.shape[0], units), dtype=np.float64)
        # array to store the downstream gradient as output of the backpropagation pipeline
        self.downstream = np.zeros_like(self.centroids, dtype=np.float64)
        # array to avoid useless memory allocation
        self.store = np.zeros(shape=(train_data.shape[0], units, train_data.shape[1]), dtype=np.float64)

    def evaluate_loss(self, labels: np.ndarray, epsilon: float = 0,
                      centroids: np.ndarray = None) -> tuple[
        float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Method evaluating the L2-penalized loss for the training data.
        :param labels: The response data, as a 1-D NumPy array.
        :param centroids: The centers took into account in the computation of the phi matrix. It is useful to avoid
        overwriting in the backtracking lineasearch pipeline.
        :param epsilon: A small value to prevent overflow issues with the exponential in the sigmoid.
        :return: A tuple with the value of the loss, the gradient vector, the phi matrix and the output of the forward
        pass.
        """
        if centroids is not None:
            phi_mat = self.interpolation_mat(train_data=self.x, centroids=centroids)
            reg = np.linalg.norm(centroids) ** 2

        else:
            phi_mat = self.interpolation_mat(train_data=self.x, centroids=self.centroids)
            reg = np.linalg.norm(self.centroids) ** 2

        out = np.dot(phi_mat, self.weights)
        out = np.squeeze(out)
        out = 1 / (1 + np.exp(-out))

        cross_entropy = -np.mean(labels * np.log(out + epsilon) + (1 - labels) * np.log(1 - (out - epsilon)))
        cross_entropy += self.rho1 * np.linalg.norm(self.weights) ** 2
        cross_entropy += self.rho2 * reg

        downstream_grad = -labels / (out + epsilon) + (1 - labels) / (1 - (out - epsilon))  # Cross-entropy gradient
        # Downstream grad for the rest of the backprop, exploiting the nice analytical shape of the sigmoid derivative
        downstream_grad = (out * (1 - out)) * downstream_grad
        gradient = self.backprop(downstream_grad[:, np.newaxis],
                                 phi_mat)  # Start the backpropagation pipeline w.r.t centers

        return cross_entropy, gradient, phi_mat, out

    def backprop(self, upstream_gradient: np.ndarray, phi_mat: np.ndarray) -> np.ndarray:
        """

        :param upstream_gradient:
        :return:
        """
        # backpropagation  w.r.t the hidden  layer
        np.dot(upstream_gradient, self.weights.T, out=self.gradient)
        np.multiply(-1 / phi_mat, self.gradient, out=self.gradient)
        np.subtract(self.x[:, np.newaxis, :], self.centroids[np.newaxis, :, :], out=self.store)
        np.multiply(self.gradient[:, :, np.newaxis], self.store, out=self.store)
        np.mean(self.store, axis=0, out=self.downstream)
        np.add(2 * self.rho2 * self.centroids, self.downstream, out=self.downstream)

        return self.downstream

    def interpolation_mat(self, train_data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """

        :param train_data:
        :param centroids:
        :return:
        """
        return np.sqrt(pairwise_distances(train_data, centroids) ** 2 + self.sigma ** 2)

    def gradient_check(self, labels: np.ndarray, epsilon: int = 1e-6):

        output_plus = np.zeros(shape=self.centroids.size, dtype=np.float64)
        centers_flatten = self.centroids.reshape(-1)
        for elem in range(len(centers_flatten)):
            current_params_plus = centers_flatten.copy()
            # adding epsilon to only one component of the entire vector of the parameters
            current_params_plus[elem] += epsilon  # adding epsilon to only one component of the entire vector of the
            # start evaluate loss pipeline
            output_plus[elem] = self.evaluate_loss(self.x, labels,
                                                   centroids=current_params_plus.reshape(self.centroids.shape),
                                                   epsilon=1e-7)[0]


        output_minus = np.zeros(shape=self.centroids.size, dtype=np.float64)
        for elem in range(len(centers_flatten)):
            current_params_minus = centers_flatten.copy()
            # subtracting epsilon to only one component of the entire vector of the parameters
            current_params_minus[elem] -= epsilon
            # start evaluate loss pipeline
            output_minus[elem] = self.evaluate_loss(self.x, labels,
                                                    centroids=current_params_minus.reshape(self.centroids.shape),
                                                    epsilon=1e-7)[0]

        grad_approx = (output_plus - output_minus) / (2 * epsilon)  # computing approximation for the gradient
        # start the pipeline to retrieve the backprop gradient
        gradient = self.evaluate_loss(self.x, labels, epsilon=1e-7,
                                      centroids=centers_flatten.reshape(self.centroids.shape))[1].reshape(-1)

        # compute the Euclidean distance normalized
        numerator = np.linalg.norm(gradient - grad_approx)
        denominator = np.linalg.norm(gradient) + np.linalg.norm(grad_approx)

        return numerator / denominator

    def fit(self, labels: np.ndarray, tol: float = 1e-4, epoch: int = 400, epsilon: float = 1e-8):

        k = 0
        conv_count = 0
        while True:
            loss, gradient, phi_mat, out = self.evaluate_loss(self.x, labels, epsilon=epsilon)

            gradient_weights = np.dot(phi_mat.T, ((out + epsilon) - labels))
            np.add(2 * self.rho1 * self.weights.reshape(-1), gradient_weights, out=gradient_weights)
            hessian_weights = phi_mat.T @ np.diag(out * (1 - out)) @ phi_mat
            hessian_weights[np.diag_indices(self.units)] += 2 * self.rho1

            np.add(self.weights, np.linalg.solve(hessian_weights, -gradient_weights)[:, np.newaxis], out=self.weights)

            alpha = self.armijo_linesearch(self.x, labels, gradient, self.centroids, epsilon=epsilon)
            self.centroids = self.centroids - alpha * gradient

            k += 1

            if np.isclose(np.linalg.norm(gradient), 0, atol=tol)\
                    and np.isclose(np.linalg.norm(gradient_weights), 0, atol=tol):
                conv_count += 1
            else:
                conv_count = 0
            if conv_count > 5 or k == epoch:
                break

        return k

    def armijo_linesearch(self, train_data: np.ndarray, labels: np.ndarray, gradient: np.ndarray, x_0: np.ndarray,
                          epsilon: float, alpha: float = 1.0, beta: float = 0.5, c1: float = 1e-3, max_iters: int = 100):

        direction = - gradient
        loss = self.evaluate_loss(train_data, labels, epsilon=epsilon)[0]

        for _ in range(max_iters):
            x_next = x_0 + alpha * direction
            loss_next = self.evaluate_loss(train_data, labels, centroids=x_next, epsilon=epsilon)[0]

            if loss <= loss_next + alpha * c1 * np.dot(gradient.reshape(-1), direction.reshape(-1)):
                break
            else:
                alpha *= beta

        return alpha

    def evaluate(self, train_data: np.ndarray):
        """

        :param train_data:
        :param labels:
        :return:
        """

        phi_mat = self.interpolation_mat(train_data, self.centroids)

        out = np.dot(phi_mat, self.weights)
        out = np.squeeze(out)
        out = 1 / (1 + np.exp(-out))

        return out

"""
if __name__ == '__main__':
    from implementation.data_import import csv_import
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np

    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(['S', 'M'], '../../data.txt', dtype=np.float64)

    x_train, x_test, y_train, y_test = train_test_split(train_data[:, :-1], train_data[:, -1], test_size=0.3)

    model = RBF(train_data=x_train, units=100, sigma=1, rho2=1e-7)

    model.fit(y_train, epoch=500)

    out = model.evaluate(x_test)

    accuracy_score(np.where(out >= 0.5, 1, 0), y_test)
"""

if __name__ == '__main__':
    from implementation.data_import import csv_import
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np

    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(['S', 'M'], '../../data.txt', dtype=np.float64, remove_dup=True)

    x_train, x_test, y_train, y_test = train_test_split(train_data[:, :-1], train_data[:, -1], test_size=0.3)

    model = RBF(train_data=x_train, units=20, sigma=1, rho1=1e-3, rho2=1e-3)
    model.gradient_check(y_train)
