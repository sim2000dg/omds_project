from time import perf_counter
import numpy as np
from sklearn.metrics import pairwise_distances
from Functions_11_DiGregorio_Castaldo import HyperTangent


class RBF:

    def __init__(self, train_data: np.ndarray, units: int = 20, rho1: int = 0,
                 rho2: int = 0, sigma: float = 1, seed: int = 1234):
        """
        :param train_data: The training data, as a NumPy array.
        :param units: The number of centers initialized in the hidden layer, as 2-D NumPy array
        :param rho1: The L2 regularization hyperparameter for weights vector. Higher rho, linearly higher L2 penalty.
        :param rho2: The L2 regularization hyperparameter for centers matrix. Higher rho, linearly higher L2 penalty.
        :param sigma: The parameter in the RBF activation function, Multiquadric RBF function is considered
        :param seed: The seed value for reproducibility
        """
        self.x = train_data
        self.rho1 = rho1
        self.rho2 = rho2
        self.sigma = sigma
        self.units = units

        self.generator = np.random.default_rng(seed)
        self.weights = self.generator.normal(0, 0.005, size=(units, 1))
        # the centers are randomly picked among the training points
        self.centroids = self.x[self.generator.choice(self.x.shape[0], units, replace=False)]

        # array to store upstream gradient in the backpropagation pipeline
        self.gradient = np.zeros(shape=(train_data.shape[0], units), dtype=np.float64)
        # array to store the downstream gradient as output of the backpropagation pipeline
        self.downstream = np.zeros_like(self.centroids, dtype=np.float64)
        # array to avoid useless memory allocation
        self.store = np.zeros(shape=(train_data.shape[0], units, train_data.shape[1]), dtype=np.float64)

    def evaluate_loss(self, train_data: np.ndarray, labels: np.ndarray, epsilon: float = 1e-8,
                      centroids: np.ndarray = None, evaluate_gradients: bool = True) -> tuple[
        float, np.ndarray, np.ndarray, np.ndarray]:
        """
        Method evaluating the L2-penalized loss for the training data.
        :param train_data: The training data, as a NumPy array.
        :param labels: The response data, as a 1-D NumPy array.
        :param centroids: The centers took into account in the computation of the phi matrix. It is useful to avoid
        overwriting in the backtracking Line-search pipeline.
        :param epsilon: A small value to prevent overflow issues with the exponential in the sigmoid.
        :param evaluate_gradients: whether evaluate the gradients
        :return: A tuple with the value of the loss, the gradient vector w.r.t the centers, the gradient vector
        w.r.t. the weights, the Hessian matrix w.r.t the weights.
        """
        # useful condition to perform gradient check and LineSearch in the fit method
        if centroids is not None:
            # Computation of the phi matrix using the Multiquadric radial basis function
            phi_mat = np.sqrt(pairwise_distances(train_data, centroids) ** 2 + self.sigma ** 2)
            reg_centroids = np.linalg.norm(centroids) ** 2  # L2 regularization w.r.t. the centers

        else:
            phi_mat = np.sqrt(pairwise_distances(train_data, self.centroids) ** 2 + self.sigma ** 2)
            reg_centroids = np.linalg.norm(self.centroids) ** 2

        out = np.dot(phi_mat, self.weights)
        out = np.squeeze(out)  # Squeeze the final output to avoid problems with broadcasting
        out = 1 / (1 + np.exp(-out))  # output for the forward pass

        # Cross entropy loss + regularization, taking into account epsilon to avoid overflow and
        # invalid arguments in Numpy.log
        cross_entropy = -np.mean(labels * np.log(out + epsilon) + (1 - labels) * np.log(1 - (out - epsilon)))
        cross_entropy += self.rho1 * np.linalg.norm(self.weights) ** 2
        cross_entropy += self.rho2 * reg_centroids

        if evaluate_gradients is False:
            return cross_entropy

        # Cross-entropy gradient, taking into account epsilon
        downstream_grad = -labels / (out + epsilon) + (1 - labels) / (1 - (out - epsilon))
        # Downstream grad for the rest of the backprop, exploiting the nice analytical shape of the sigmoid derivative
        downstream_grad = (out * (1 - out)) * downstream_grad

        # Start the backpropagation pipeline w.r.t centers
        gradient_centroids = self.backprop(downstream_grad[:, np.newaxis],
                                           phi_mat)

        # Analytic gradient w.r.t the weights vector
        gradient_weights = np.dot(phi_mat.T, ((out + epsilon) - labels))
        np.add(2 * self.rho1 * self.weights.reshape(-1), gradient_weights, out=gradient_weights)

        # Analytic hessian w.r.t the weights vector
        hessian_weights = phi_mat.T @ np.diag(out * (1 - out)) @ phi_mat
        hessian_weights[np.diag_indices(self.units)] += 2 * self.rho1

        return cross_entropy, gradient_centroids, gradient_weights, hessian_weights

    def backprop(self, upstream_gradient: np.ndarray, phi_mat: np.ndarray) -> np.ndarray:

        """
        Returns the gradient for the hidden layer w.r.t the centers matrix
        :param upstream_gradient: The upstream gradient as a 2-D NumPy array
        :param phi_mat: The design matrix transformed as a 2-D Numpy array
        :return: The downstream gradient w.r.t the centers as a 2-D Numpy array
        """
        # save gradient w.r.t the phi matrix
        np.dot(upstream_gradient, self.weights.T, out=self.gradient)
        # save the element-wise multiplication between the derivative of the RBF function and the upstream gradient
        np.multiply(-1 / phi_mat, self.gradient, out=self.gradient)

        # the derivative of the norm is a 3D tensor with the suitable dimensions
        np.subtract(self.x[:, np.newaxis, :], self.centroids[np.newaxis, :, :], out=self.store)
        # save the element-wise multiplication between the upstream gradient expanded and the derivative of the norm
        np.multiply(self.gradient[:, :, np.newaxis], self.store, out=self.store)

        # Save the average gradient
        np.mean(self.store, axis=0, out=self.downstream)

        # Simply adding the gradient of the L2 regularization to the downstream gradient
        gradient = np.add(2 * self.rho2 * self.centroids, self.downstream, out=self.downstream)

        return gradient

    def gradient_check(self, labels: np.ndarray, epsilon: int = 1e-6) -> str:
        """
        :param labels: The response data, as a 1-D NumPy array.
        :param epsilon: Small constant for the perturbation of the parameters
        :return A string with the result of the gradient check
        """

        output_plus = np.zeros(shape=self.centroids.size, dtype=np.float64)
        centers_flatten = self.centroids.reshape(-1)
        for elem in range(len(centers_flatten)):
            current_params_plus = centers_flatten.copy()
            # adding epsilon to only one component of the entire vector of the parameters
            current_params_plus[elem] += epsilon  # adding epsilon to only one component of the entire vector of the
            # start evaluate loss pipeline
            output_plus[elem] = self.evaluate_loss(self.x, labels, centroids=current_params_plus.reshape(self.centroids.shape),
                                                   evaluate_gradients=False)

        output_minus = np.zeros(shape=self.centroids.size, dtype=np.float64)
        for elem in range(len(centers_flatten)):
            current_params_minus = centers_flatten.copy()
            # subtracting epsilon to only one component of the entire vector of the parameters
            current_params_minus[elem] -= epsilon
            # start evaluate loss pipeline
            output_minus[elem] = self.evaluate_loss(self.x, labels,
                                                    centroids=current_params_minus.reshape(self.centroids.shape),
                                                    evaluate_gradients=False)

        grad_approx = (output_plus - output_minus) / (2 * epsilon)  # computing approximation for the gradient
        # start the pipeline to retrieve the backprop gradient
        gradient = self.evaluate_loss(self.x, labels, epsilon=1e-7,
                                      centroids=centers_flatten.reshape(self.centroids.shape))[1].reshape(-1)

        # compute the Euclidean distance normalized
        numerator = np.linalg.norm(gradient - grad_approx)
        denominator = np.linalg.norm(gradient) + np.linalg.norm(grad_approx)

        if numerator / denominator <= epsilon:
            print('The analytic gradient is correct !! The norm of the difference between the gradient approximation ' \
                  f'and the actual gradient is {numerator:09}')
        else:
            print(
                'The analytic gradient is  not correct !! The norm of the difference between the gradient approximation ' \
                f'and the actual gradient is {numerator :09}')

    def fit(self, train_data: np.ndarray, labels: np.ndarray, tol: float = 1e-4, epoch: int = 400,
            early_stopping: int = 5) -> dict:
        """
        "The fit method implements the 2-blocks decomposition algorithm. The weights vector is updated using
        the Newton-Raphson algorithm, where a single update is determined by evaluating the gradient and
        the Hessian matrix.
        After updating the weights, the centers are updated through a backtracking line search (Armijo) to determine
        the distance to move along the steepest descent direction.
        If the global optimum for the weights exists, by the convexity of the objective function, this algorithm
        converges to the global optimum, while the global minimum for the centers is not guaranteed.
        However, every sequence {(w_k),(c_k)} admits an accumulation point, {E(w_k),(c_k)} converges and every
        accumulation point of {(w_k , c_k )} is a stationary point."

        :param train_data: The training data, as a NumPy array.
        :param labels: The response data, as a 1-D NumPy array.
        :param tol: A small constant for the stopping criterion
        :param epoch: The max number of iteration performed
        :param early_stopping: The maximum number of iterations allowed without any improvement.
        :return The number of iteration, the value of the loss and the reason for stopping.
        """
        k = 0  # Iterations of the algorithm
        n_eval = 0  # Number of evaluation of the loss function
        conv_count = 0  # Counter for the stopping criterion condition
        es_counter = 0  # Counter for the early stopping criterion condition
        loss_last = None  # save the loss value at the previous iteration
        loss_init = None  # save the initial training loss
        start = perf_counter()  # Start the time counter to optimize the network

        while True:
            loss, gradient_centroids, gradient_weights, hessian_weights = self.evaluate_loss(train_data, labels)
            if loss_init is None:
                loss_init = loss
            n_eval += 1
            loss_last = loss

            # Update of the weights vector
            try:
                np.add(self.weights, np.linalg.solve(hessian_weights, - gradient_weights)[:, np.newaxis],
                       out=self.weights)
                k += 1
            except np.linalg.LinAlgError:
                raise ValueError('Regularization on weights vector is too low (i.e. the outputs of the RBFs are '
                                 'linearly dependent)')

            loss, gradient_centroids = self.evaluate_loss(train_data, labels)[0:2]
            n_eval += 1
            # line search to find the optimal step size
            alpha, k_armijo = self.armijo_linesearch(train_data, labels, gradient_centroids, self.centroids, loss)
            n_eval += k_armijo
            # Update centers along the steepest descent direction
            self.centroids = self.centroids - alpha * gradient_centroids
            k += 1

            # Condition for early stopping criterion
            es_counter = es_counter + 1 if (
                    np.greater_equal(loss, loss_last) | np.isclose(loss, loss_last, atol=1e-4)) else 0

            # Condition for the stopping criterion of the algorithm
            conv_count = conv_count + 1 if (
                    np.isclose(np.linalg.norm(gradient_centroids), 0, atol=tol)
                    and np.isclose(np.linalg.norm(gradient_weights), 0, atol=tol)) \
                else 0

            if conv_count > 5 or k == epoch or es_counter == early_stopping:
                break

        end = perf_counter()  # Stop the time counter

        return dict(n_iter=k,
                    gradient_evals=k,
                    fun_evals=n_eval,
                    fun_init=loss_init,
                    fun=loss,
                    time=round(end - start, 3),
                    early_stopping='Early Stopping ...' if es_counter == early_stopping
                    else 'The optimization routine was not early stopped',
                    message=print(f'Training completed in {k} iterations') if conv_count != 5
                    else print(f'convergence reached in {k} iterations'),
                    success=True)

    def armijo_linesearch(self, train_data: np.ndarray, labels: np.ndarray, gradient: np.ndarray, x_0: np.ndarray,
                          loss: float,
                          alpha: float = 1.0, beta: float = 0.5, c1: float = 1e-3,
                          max_iters: int = 50) -> tuple[float, int]:
        """
        Performing Armijo line search to determine the amount to move along a given search direction
        :param train_data:
        :param labels: The response data, as a 1-D NumPy array.
        :param gradient: direction for the search
        :param x_0: The staring point
        :param loss: The loss value after the update of the weights vector
        :param alpha: The maximum candidate step size
        :param beta: The search control parameter
        :param c1: The parameter to compute the loss at the next step
        :param max_iters: The maximum number of iterations allowed
        :return The learning rate alpha
        """

        direction = - gradient
        k = 0  # number of evaluations of the loss function
        while True:
            x_next = x_0 + alpha * direction
            loss_next = self.evaluate_loss(train_data, labels, centroids=x_next, evaluate_gradients=False)
            k += 1

            if loss <= loss_next + alpha * c1 * np.dot(gradient.reshape(-1), direction.reshape(-1)) or k == max_iters:
                break
            else:
                alpha *= beta

        return alpha, k

    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """
        Method returning a 1-D array of predictions.
        :params test_data: The array (compatible with the initialized and trained model) containing the test data.
        :returns: The 1-D array of predictions.
        """

        phi_mat = np.sqrt(pairwise_distances(test_data, self.centroids) ** 2 + self.sigma ** 2)

        out = np.dot(phi_mat, self.weights)
        out = np.squeeze(out)
        out = 1 / (1 + np.exp(-out))

        return out

