import numpy as np
from abc import abstractmethod
from typing import Self


# Credits to Prof. Galasso's slides for the matrix/tensor approach to backpropagation


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
        self.downstream = None
        self.initialized = False

    def _forward(self, input_array: np.array):
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

    def _backprop(self, upstream_gradient):
        """Returns the downstream tensor gradient for this layer, considering the whole
        set of observations in the forward pass
        :return: The downstream gradient
        """
        # Save avg. gradient w.r.t. current weights
        np.dot(self.back_store.T, upstream_gradient, out=self.tensor_grad)
        np.mean(self.tensor_grad, axis=0, out=self.gradient)

        # Return downstream gradient
        return np.dot(upstream_gradient, np.transpose(self.weights), out=self.downstream)

    def _model_setup(self, batch_size, in_shape) -> Self:
        """
        Method called when connecting this module to a Model object
        :param batch_size: The batch size for the model
        :return:
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

    def _forward(self, input_array):
        if not self.initialized:
            raise AttributeError('Your layer is not inside a model, and therefore not initialized')
        self.back_store = input_array
        np.multiply(input_array, 2 * self.sigma, out=self.out)
        np.exp(self.out, out=self.out)
        np.add(self.out, 1, out=self.denom)
        np.add(self.out, -1, out=self.out)
        np.divide(self.out, self.denom, out=self.out)

        return self.out

    def _backprop(self, upstream_gradient):
        """
        Returns the gradient for this layer, considering
        the whole set of observations in the forward pass
        :param upstream_gradient: A NumPy array giving the upstream gradient w.r.t. the activation function
        """
        # Save (exp(2sigma*x)+1)^2
        np.square(self.denom, out=self.downstream)
        # Save 4*sigma*exp(2sigma*x)
        np.subtract(self.denom, 1, out=self.denom)
        np.multiply(4*self.sigma, self.denom, out=self.denom)
        # Save and return the ratio
        np.divide(self.denom, self.downstream, out=self.downstream)

        return self.downstream

    def _model_setup(self, batch_size, in_shape) -> Self:
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


class Model:
    """
    A class representing the whole model
    """

    def __init__(self):
        self.layers = list()

    def clean_back(self):
        for layer in self.layers:
            del layer.gradie

    def evaluate_loss(self, train_data: np.array, labels: np.array, current_params: np.array) -> Self:
        """
        Method evaluating the L2-penalized loss for the training data
        :param train_data: The training data, as a NumPy array
        :param labels: The response data, as a 1-D NumPy array
        :param current_params: The parameters of the network in a 1-D NumPy array
        :return: The value of the loss
        """
        return self

    def add(self, layer: Linear | HyperTangent) -> Self:
        """
        Method adding a layer to the model
        :param layer: The layer object, initialized
        :return: The object itself
        """

    def backprop(self, *args, **kwargs) -> np.array:
        """
        Method starting the backpropagation chain in order to get the overall gradient out
        :returns: The n-D gradient array, where n is the number of parameters in the network
        """

    def evaluate(self, test_data: np.array) -> np.array:
        """
        Method returning a 1-D array of predictions
        :params test_data: The array (compatible with the initialized and trained model) containing the test data
        :returns: The 1-D array of predictions
        """
