import numpy as np
from typing import Self
# from .data_import import csv_import
import math

# Credits to Prof. Galasso's slides for guidelines for efficient backpropagation


class Linear:
    """
    A class representing a dense layer with given number of inputs and outputs
    """

    def __init__(self, units):
        self.units = units
        self.batch_size = None
        self.back_store = None
        self.in_shape = None
        self.weights = None
        self.bias = None
        self.gradient = None
        self.store = None  # Array used just to avoid useless data copying and array allocation
        self.out = None
        self.tensor_grad = None
        self.gradient_bias = None
        self.downstream = None
        self.initialized = False

    def forward(self, input_array: np.array):
        """
        Forward pass for Linear layer
        :param input_array: A (batch_size x in_shape) array
        :return: The output of the forward pass for this linear layer
        """
        if not self.initialized:
            raise AttributeError('Your layer is not inside a model, and therefore not initialized')
        self.back_store = input_array  # Save the input array for backpropagation

        np.dot(input_array, self.weights, out=self.store)  # Matrix multiplication
        np.add(self.store, self.bias, out=self.out)  # Adding bias
        return self.out  # Return output of forward pass

    def backprop(self, upstream_gradient):
        """Returns the downstream tensor gradient for this layer, considering the whole
        set of observations in the forward pass
        :return: The downstream gradient
        """
        # Save avg. gradient w.r.t. current weights
        np.dot(self.back_store.T, upstream_gradient, out=self.tensor_grad)
        np.mean(self.tensor_grad, axis=0, out=self.gradient)
        np.add(self.gradient, self.weights, out=self.gradient)
        np.add(self.gradient, self.weights, out=self.gradient)

        self.gradient_bias = upstream_gradient

        # Return downstream gradient
        return np.dot(upstream_gradient, np.transpose(self.weights), out=self.downstream)

    def model_setup(self, batch_size: int, in_shape: int) -> Self:
        """
        Method called when connecting this module to a Model object
        :param batch_size: The batch size for the model
        :param in_shape: Input shape for the layer
        :return: The object itself
        """
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.weights = np.zeros((in_shape, self.units), dtype=np.float32)
        self.bias = np.zeros(shape=self.units, dtype=np.float32)
        self.gradient = np.zeros((in_shape, self.units), dtype=np.float32)
        self.out = np.zeros(shape=(self.batch_size, self.units), dtype=np.float32)
        self.tensor_grad = np.zeros((self.batch_size, self.in_shape, self.units), dtype=np.float32)
        self.downstream = np.zeroes(self.batch_size, self.in_shape, dtype=np.float32)
        self.store = np.zeros(shape=(self.batch_size, self.units), dtype=np.float32)
        self.initialized = True

        return self

    def __call__(self, input):
        return self.forward(input)


class HyperTangent:
    """
    A class representing a Hyperbolic Tangent activation, applied element-wise.
    """

    def __init__(self, sigma):
        """
        Initialization of Hyperbolic Tangent Activation object
        :param sigma: The sigma chosen for the activation
        """
        if sigma <= 0:
            raise ValueError('The value of sigma must be greater than 0')
        self.sigma = sigma
        self.in_shape = None
        self.batch_size = None
        self.back_store = None
        self.out = None
        self.denom = None  # Forward pass denominator
        self.downstream = None
        self.initialized = False

    def forward(self, input_array):
        if not self.initialized:
            raise AttributeError('Your layer is not inside a model, and therefore not initialized')
        self.back_store = input_array
        np.multiply(input_array, 2 * self.sigma, out=self.out)
        np.exp(self.out, out=self.out)
        np.add(self.out, 1, out=self.denom)
        np.add(self.out, -1, out=self.out)
        np.divide(self.out, self.denom, out=self.out)

        return self.out

    def backprop(self, upstream_gradient):
        """
        Returns the gradient for this layer, considering
        the whole set of observations in the forward pass
        :param upstream_gradient: A NumPy array giving the upstream gradient w.r.t. the activation function
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

    def model_setup(self, batch_size, in_shape) -> Self:
        """
        Method called when connecting this module to a Model object
        :param batch_size: The batch size for the model
        :return: The object itself
        """
        self.batch_size = batch_size
        self.in_shape = in_shape
        self.out = np.zeros(self.batch_size, self.in_shape, dtype=np.float32)
        self.downstream = np.zeroes(self.batch_size, self.in_shape, dtype=np.float32)
        self.denom = np.zeroes(self.batch_size, self.in_shape, dtype=np.float32)
        self.initialized = True
        return self

    def __call__(self, input):
        return self.forward(input)


class Model:
    """
    A class representing the whole MLP model
    """

    def __init__(self, batch_size, input_shape, rho):
        self.current_out_shape = input_shape
        self.batch_size = batch_size
        self.layers: list[Linear | HyperTangent, ...] = list()
        self.rho: float = rho  # L2 regularization

    def add(self, layer: Linear | HyperTangent) -> Self:
        """
        Method adding a layer to the model
        :param layer: The layer object, initialized
        :return: The object itself
        """
        self.layers.append(layer.model_setup(batch_size=self.batch_size,
                                             in_shape=self.current_out_shape))
        return self

    def backprop(self, upstream_gradient) -> np.array:
        """
        Method starting the backpropagation chain in order to get the overall gradient out
        :returns: The n-D gradient array, where n is the number of parameters in the network
        """
        gradient_list = list()
        for layer in reversed(self.layers):
            upstream_gradient = layer.backprop(upstream_gradient)
            if isinstance(layer, Linear):
                local_gradient = layer.gradient
                gradient_bias = layer.gradient_bias
                gradient_list.extend([local_gradient, gradient_bias])

        gradient_list = [np.reshape(x, -1) for x in reversed(gradient_list)]
        gradient = np.concatenate(gradient_list)

        return gradient

    def evaluate_loss(self, train_data: np.array, labels: np.array, current_params: np.array) -> tuple[float, np.array]:
        """
        Method evaluating the L2-penalized loss for the training data. It returns the value of the loss and the gradient
        :param train_data: The training data, as a NumPy array
        :param labels: The response data, as a 1-D NumPy array
        :param current_params: The parameters of the network in a 1-D NumPy array
        :return: A tuple with the value of the loss and the gradient vector
        """
        for layer in self.layers:
            pos = 0
            if isinstance(layer, Linear):
                slice_dim_bias = layer.bias.shape
                slice_dim_weights = math.prod(layer.weights.shape)
                layer.bias[:] = current_params[pos: pos+slice_dim_bias]
                pos += slice_dim_bias
                layer.weights.flat[:] = current_params[pos: pos+slice_dim_weights]
                pos += slice_dim_weights

        out = train_data
        reg = 0
        for layer in self.layers:
            out = layer(out)  # Computing the forward pass we are
            # also storing information necessary for the backward pass
            reg += np.linalg.norm(layer.weights)

        out = 1 / (1 + np.exp(out))  # To be upgraded if we want
        cross_entropy = np.mean(labels * np.log(out) + 1 - labels * np.log(1 - out)) + self.rho * reg

        downstream_grad = -labels / out + (1 - labels) / (1 - out)  # Cross-entropy gradient
        # Downstream grad for the rest of the backprop, exploiting the nice analytical shape of the sigmoid derivative
        downstream_grad = (out * (1-out))*downstream_grad
        gradient = self.backprop(downstream_grad)

        return cross_entropy, gradient

    def evaluate(self, test_data: np.array) -> np.array:
        """
        Method returning a 1-D array of predictions
        :params test_data: The array (compatible with the initialized and trained model) containing the test data
        :returns: The 1-D array of predictions
        """
        pass


if __name__ == '__main__':
    from data_import import csv_import

    train_data = csv_import(['S', 'M'], '../data.txt')
    model = Model(64, 16)
    model.add(Linear(2))
    model.add(HyperTangent(0.5))
    model.evaluate_loss(train_data[:, :-1], train_data[:, -1])
