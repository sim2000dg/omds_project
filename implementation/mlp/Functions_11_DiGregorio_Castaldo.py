import numpy as np
from typing import Optional, Iterator
from scipy.optimize import OptimizeResult
import math
from time import perf_counter


# Credits to Prof. Galasso's slides for guidelines for efficient backpropagation


class Linear:
    """
    The dense layer of a MLP deep learning architecture with given number of inputs and outputs.
    """

    def __init__(self, units: int):
        """
        Initialization method of the Linear layer.
        :param units: The number of so-called hidden units. The output dimensionality.
        """
        self.units = units
        self.batch_size: Optional[int] = None
        self.back_store: Optional[np.ndarray] = None
        self.rho: Optional[float] = None
        self.in_shape: Optional[int] = None
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.gradient: Optional[np.ndarray] = None
        self.gradient_bias: Optional[np.ndarray] = None
        self.back_reg: Optional[
            np.ndarray
        ] = None  # Array used just to avoid useless memory allocation
        self.store: Optional[
            np.ndarray
        ] = None  # Array used just to avoid useless memory allocation
        self.out: Optional[np.ndarray] = None
        self.downstream: Optional[np.ndarray] = None
        self.initialized = False

    def forward(self, input_array: np.ndarray):
        """
        Forward pass for the Linear layer, a matrix multiplication with bias addition.
        :param input_array: A (batch_size x in_shape) NumPy array.
        :return: The output of the forward pass for this linear layer.
        """
        if not self.initialized:
            raise AttributeError(
                "Your layer is not inside a model, and therefore not initialized"
            )
        self.back_store = input_array  # Save the input array for backpropagation

        np.dot(input_array, self.weights, out=self.store)  # Matrix multiplication
        np.add(self.store, self.bias, out=self.out)  # Adding bias
        return self.out  # Return output of forward pass

    def backprop(self, upstream_gradient):
        """Returns the downstream gradient for this layer and computes the gradient for the layer parameters,
        saving it in a pre-allocated array to be accessed with the complete backpropagation pipeline.
        :return: The downstream gradient
        """
        # Save avg. gradient w.r.t. current weights
        np.dot(self.back_store.T, upstream_gradient, out=self.gradient)
        np.divide(self.gradient, self.batch_size, out=self.gradient)
        # Adding regularization part of the gradient
        np.multiply(2 * self.rho, self.weights, out=self.back_reg)
        np.add(self.gradient, self.back_reg, out=self.gradient)

        # The gradient for the bias is simply the upstream gradient. We need to take the average over batch axis
        np.mean(upstream_gradient, axis=0, out=self.gradient_bias)

        # Return downstream gradient
        np.dot(upstream_gradient, np.transpose(self.weights), out=self.downstream)
        return self.downstream

    def model_setup(self, batch_size: int, in_shape: int, rho: float) -> 'Self':
        """
        Method called when connecting this module to a Model object. This method initializes the module and the
        necessary arrays and attributes needed for the forward and the backward pass.
        :param batch_size: The batch size for the model.
        :param in_shape: Input shape for the layer.
        :param rho: L2 regularization hyperparameter.
        :return: The object itself.
        """
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.weights = np.zeros((in_shape, self.units), dtype=np.float64)
        self.bias = np.zeros(shape=self.units, dtype=np.float64)
        self.gradient_bias = np.full_like(self.bias, 0)
        self.gradient = np.full_like(self.weights, 0)
        self.out = np.zeros((self.batch_size, self.units), dtype=np.float64)
        self.downstream = np.zeros((self.batch_size, self.in_shape), dtype=np.float64)
        self.store = np.full_like(self.out, 0)
        self.back_reg = np.full_like(self.weights, 0)
        self.rho = rho
        self.initialized = True

        return self

    def __call__(self, input):  # call should be just forward
        return self.forward(input)


class HyperTangent:
    """
    The layer for the hyperbolic tangent activation, applied element-wise.
    """

    def __init__(self, sigma: float):
        """
        Initialization of the layer for the hyperbolic tangent activation.
        :param sigma: The sigma chosen for the activation, a dispersion parameter for the hyperbolic tangent.
        """
        if sigma <= 0:
            raise ValueError("The value of sigma must be greater than 0")
        self.sigma = sigma
        self.in_shape: Optional[int] = None
        self.batch_size: Optional[int] = None
        self.back_store: Optional[np.ndarray] = None
        self.out: Optional[np.ndarray] = None
        self.denom: Optional[np.ndarray] = None  # Forward pass denominator
        self.downstream: Optional[np.ndarray] = None
        self.initialized = False

    def forward(self, input_array):
        """
        Forward pass for the hyperbolic tangent activation layer.
        :param input_array: A (batch_size x in_shape) NumPy array.
        :return: The output of the forward pass for this layer.
        """
        if not self.initialized:
            raise AttributeError(
                "Your layer is not inside a model, and therefore not initialized"
            )
        self.back_store = input_array
        np.multiply(input_array, 2 * self.sigma, out=self.out)
        np.exp(self.out, out=self.out)
        np.add(self.out, 1, out=self.denom)
        np.add(self.out, -1, out=self.out)
        np.divide(self.out, self.denom, out=self.out)

        return self.out

    def backprop(self, upstream_gradient):
        """
        An activation function has no trained parameter, so this method only computes and propagates downstream the
        downstream gradient.
        :param upstream_gradient: A NumPy array giving the gradient of the loss (already summed over the batch axis)
         w.r.t. outputs of the activation function layer.
        """
        # Save (exp(2sigma*x)+1)^2
        np.square(self.denom, out=self.downstream)
        # Save 4*sigma*exp(2sigma*x)
        np.subtract(self.denom, 1, out=self.denom)
        np.multiply(4 * self.sigma, self.denom, out=self.denom)
        # Save and return the ratio
        np.divide(self.denom, self.downstream, out=self.downstream)
        # Compute actual downstream gradient
        np.multiply(self.downstream, upstream_gradient, out=self.downstream)

        return self.downstream

    def model_setup(self, batch_size, in_shape, **kwargs) -> 'Self':
        """
        Method called when connecting this module to a Model object. This method initializes the module and the
        necessary arrays and attributes needed for the forward and the backward pass.
        :param batch_size: The batch size for the model.
        :param in_shape: Input shape for the layer.
        :return: The object itself
        """
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out = np.zeros((self.batch_size, self.in_shape), dtype=np.float64)
        self.downstream = np.full_like(self.out, 0)
        self.denom = np.full_like(self.out, 0)
        self.initialized = True
        return self

    def __call__(self, input):  # call should be just forward
        return self.forward(input)


class Model:
    """
    A class representing the whole MLP model
    """

    def __init__(self, batch_size: int, input_shape: int, rho: float = 0):
        """
        Initialization method for MLP Model class.
        :param batch_size: The batch size for the MLP model. It needs to be fixed for efficiency reasons (no spare obs.)
        :param input_shape: The input shape for this model, the dimensionality of the input observations.
        :param rho: The L2 regularization hyperparameter. Higher rho, linearly higher L2 penalty.
        """
        self.current_out_shape = input_shape
        self.batch_size = batch_size
        self.layers: list[Linear | HyperTangent, ...] = list()
        self.rho = rho  # L2 regularization
        self.num_obs: Optional[int] = None
        self.generator: Optional[np.random._generator.Generator] = None
        self.epochs: Optional[int] = None
        self.yielder: Optional[Iterator] = None
        self.completed_train: bool = False
        self.tot_params: int = 0
        self.gradient_vector: Optional[np.ndarray] = None

    def add(self, layer: Linear | HyperTangent) -> 'Self':
        """
        Method adding a layer to the model. The method also triggers layer initialization.
        :param layer: The layer object.
        :return: The object itself.
        """
        # Trigger initialization with batch size and input shape
        self.layers.append(
            layer.model_setup(
                batch_size=self.batch_size,
                in_shape=self.current_out_shape,
                rho=self.rho,
            )
        )
        if isinstance(
            layer, Linear
        ):  # Save total number of parameters for arrays pre-allocation
            self.tot_params += math.prod(layer.weights.shape)
            self.tot_params += layer.units
        # Set current output shape for the model
        self.current_out_shape = layer.out.shape[-1]
        return self

    def backprop(self, upstream_gradient) -> np.ndarray:
        """
        Backpropagation pipeline for the overall MLP model.
        :returns: The n-D gradient array, where n is the number of parameters in the network.
        """
        gradient_list = list()  # List of gradient arrays
        for layer in reversed(self.layers):  # Go backwards in list of layers
            upstream_gradient = layer.backprop(
                upstream_gradient
            )  # Get upstream gradient for following layer
            if isinstance(
                layer, Linear
            ):  # If layer is linear, layer.backprop also computes parameters gradient
                local_gradient = layer.gradient  # Pointer to current parameters gradient
                gradient_bias = layer.gradient_bias  # Pointers to current bias gradient
                gradient_list.extend(
                    [local_gradient, gradient_bias]
                )  # Extend gradient list with two pointers

        # Flatten everything in C-contiguous ordering
        gradient_list = [np.reshape(x, -1) for x in reversed(gradient_list)]
        np.concatenate(
            gradient_list, out=self.gradient_vector
        )  # Concatenate the gradient vectors in one long gradient vector

        return self.gradient_vector

    def evaluate_loss(
        self,
        current_params: np.ndarray,
        train_data: np.ndarray,
        labels: np.ndarray,
        epochs: int,
        seed: int,
    ) -> tuple[float, np.ndarray]:
        """
        Method evaluating the L2-penalized loss for the training data.
        It returns the value of the loss and the gradient.
        :param current_params: The **ordered** parameters of the network in a 1-D NumPy array.
        :param train_data: The training data, as a NumPy array.
        :param labels: The response data, as a 1-D NumPy array.
        :param seed: Seed for reproducible training.
        :param epochs: Number of epochs for training.
        :return: A tuple with the value of the loss and the gradient vector.
        """
        if self.layers[-1].out.shape[-1] != 1:
            raise ValueError(
                "The last layer needs to have just one neuron, since it is the output one "
                "and the output is scalar valued"
            )

        if self.num_obs is None:
            self.num_obs = len(train_data)
        if self.generator is None:
            self.generator = np.random.default_rng(seed)
        if self.epochs is None:
            self.epochs = epochs
        if self.gradient_vector is None:
            self.gradient_vector = np.zeros(self.tot_params, dtype=np.float64)

        # Setting the parameters in the net
        pos = 0  # Variable storing the slice position
        for layer in self.layers:
            if isinstance(
                layer, Linear
            ):  # We have parameters only for the Linear layer
                if (
                    len(current_params) < pos + 1
                ):  # Check that number of passed parameters is enough
                    raise ValueError(
                        "The length of the current_params array is not enough to cover "
                        "the number of parameters"
                    )
                slice_dim_bias = layer.bias.shape[0]  # Get bias dimension
                slice_dim_weights = math.prod(
                    layer.weights.shape
                )  # Get number of weights in weight matrix
                layer.bias[:] = current_params[
                    pos : pos + slice_dim_bias
                ]  # Set params in pre-allocated bias array
                pos += slice_dim_bias  # Update slice position w.r.t. input 1-D params array
                layer.weights.flat[:] = current_params[
                    pos : pos + slice_dim_weights
                ]  # Set params in weights array
                pos += slice_dim_weights  # Update slice position w.r.t. input 1-D params array

        if self.yielder is None:
            self.yielder = self.batch_yielder(train_data, labels).__iter__()

        out, labels, it = next(self.yielder)  # Initialize layer input and labels
        if it == self.epochs * (len(train_data) // self.batch_size):
            self.completed_train = True
        reg = 0  # Initialize L2 penalty
        for layer in self.layers:  # Forward pass, layer per layer
            out = layer(
                out
            )  # Computing this pass we are also storing info necessary for the backward pass
            if isinstance(
                layer, Linear
            ):  # If layer is linear, we add to the penalty the norm of the weight matrix
                reg += np.linalg.norm(layer.weights) ** 2

        # Epsilon to prevent problems with the logarithm in the cross-entropy
        epsilon = 1e-6
        out = np.squeeze(
            out
        )  # Squeeze the final output to avoid problems with broadcasting
        out = 1 / (1 + np.exp(-out))  # Apply sigmoid activation
        # Cross entropy loss + regularization
        cross_entropy = (
            -np.mean(
                labels * np.log(out + epsilon)
                + (1 - labels) * np.log(1 - (out - epsilon))
            )
            + self.rho * reg
        )

        downstream_grad = -labels / (out + epsilon) + (1 - labels) / (
            1 - (out - epsilon)
        )  # Cross-entropy gradient, taking into account epsilon
        # Downstream grad for the rest of the backprop, exploiting the nice analytical shape of the sigmoid derivative
        downstream_grad = (out * (1 - out)) * downstream_grad
        gradient = self.backprop(
            downstream_grad[:, np.newaxis]
        )  # Start the backpropagation pipeline

        return cross_entropy, gradient, self.completed_train

    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """
        Method returning a 1-D array of predictions.
        :params test_data: The array (compatible with the initialized and trained model) containing the test data.
        :returns: The 1-D array of predictions.
        """
        # This method is not really optimized for performance, but evaluation for prediction is far less problematic
        # than the one for training, since just one pass is required.
        output = test_data
        for layer in self.layers:
            if isinstance(layer, Linear):
                output = output @ layer.weights
                output += layer.bias
            elif isinstance(layer, HyperTangent):
                output = np.exp(layer.sigma * output)
                output = (output - 1) / (output + 1)

        output = 1 / (1 + np.exp(-output))
        return output

    def batch_yielder(self, train_data, labels):
        num_batch = len(train_data) // self.batch_size
        it = 0
        for _ in range(self.epochs):
            pos = 0
            shuffle_idx = self.generator.permutation(len(train_data))
            train_data = train_data[shuffle_idx]
            labels = labels[shuffle_idx]
            for batch in range(num_batch):
                it += 1
                yield train_data[pos : pos + self.batch_size], labels[
                    pos : pos + self.batch_size
                ], it

    def gradient_check(
        self,
        train_data: np.array,
        labels: np.array,
        current_params: np.array,
        epsilon: float = 1e-6,
    ) -> str:
        """
        Method returning the Euclidean distance between the numeric gradient and the output from the backprop pipeline,
        normalized by the sum of the norms of the vectors loss for the training data.

        :param train_data: The training data, as a NumPy array.
        :param labels: The response data, as a 1-D NumPy array.
        :param current_params: The **ordered** parameters of the network in a 1-D NumPy array.
        :param epsilon: Small constant.
        :return The Euclidean distance between the numeric gradient and the backprop gradient.
        """

        output_plus = np.zeros(current_params.shape[0], dtype=np.float64)
        for elem in range(len(current_params)):
            current_params_plus = current_params.copy()
            # adding epsilon to only one component of the entire vector of the parameters
            current_params_plus[
                elem
            ] += epsilon  # adding epsilon to only one component of the entire vector of the
            # start evaluate loss pipeline
            output_plus[elem] = self.evaluate_loss(current_params_plus,
                train_data, labels, seed=1234, epochs=1
            )[0]
            self.yielder = None

        output_minus = np.zeros(current_params.shape[0], dtype=np.float64)
        for elem in range(len(current_params)):
            current_params_minus = current_params.copy()
            # subtracting epsilon to only one component of the entire vector of the parameters
            current_params_minus[elem] -= epsilon
            # start evaluate loss pipeline
            output_minus[elem] = self.evaluate_loss(current_params_minus,
                train_data, labels, epochs=1, seed=1234
            )[0]
            self.yielder = None

        grad_approx = (output_plus - output_minus) / (
            2 * epsilon
        )  # computing approximation for the gradient
        # start the pipeline to retrieve the backprop gradient
        gradient = self.evaluate_loss(current_params, train_data, labels, seed=1234, epochs=1)[1]
        self.yielder = None

        # compute the Euclidean distance normalized
        numerator = np.linalg.norm(gradient - grad_approx)
        denominator = np.linalg.norm(gradient) + np.linalg.norm(grad_approx)
        return numerator / denominator


def Adam_sciPy(fun, x0, args, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
    m = np.zeros_like(x0)
    s = np.zeros_like(x0)
    i = 0
    loss_init = None
    start = perf_counter()
    while True:
        loss, gradient, status = fun(x0, *args)
        if loss_init is None:
            loss_init = loss
        m *= beta1
        m += (1 - beta1) * gradient
        s *= beta2
        s += (1 - beta2) * gradient ** 2
        mhat = m / (1 - beta1 ** (i + 1))
        vhat = s / (1 - beta2 ** (i + 1))
        x0 -= lr * mhat / (np.sqrt(vhat) + eps)
        i += 1

        if status is True:
            break

    end = perf_counter()

    return OptimizeResult(
        x=x0,
        fun=loss,
        fun_init=loss_init,
        jac=gradient,
        nit=i if status is True else i - 1,
        time=round(end-start, 3),
        message="Training completed" if status is True else "Convergence reached",
        success=True,
    )
