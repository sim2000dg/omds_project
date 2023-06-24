import numpy as np
from typing import Self
# from .data_import import csv_import
import math
from typing import Optional

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
        self.back_reg: Optional[np.ndarray] = None  # Array used just to avoid useless memory allocation
        self.store: Optional[np.ndarray] = None  # Array used just to avoid useless memory allocation
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
            raise AttributeError('Your layer is not inside a model, and therefore not initialized')
        self.back_store = input_array  # Save the input array for backpropagation

        np.dot(input_array, self.weights, out=self.store)  # Matrix multiplication
        np.add(self.store, self.bias, out=self.out)  # Adding bias
        return self.out  # Return output of forward pass

    def backprop(self, upstream_gradient):
        """Returns the downstream gradient for this layer and computes the local gradient for the layer,
        saving it in a pre-allocated array to be accessed with the complete backpropagation pipeline.
        :return: The downstream gradient
        """
        # Save avg. gradient w.r.t. current weights
        np.dot(self.back_store.T, upstream_gradient, out=self.gradient)
        np.divide(self.gradient, self.batch_size, out=self.gradient)
        # Adding regularization part of the gradient
        np.multiply(2*self.rho, self.weights, out=self.back_reg)
        np.add(self.gradient, self.back_reg, out=self.gradient)

        # The gradient for the bias is simply the upstream gradient. We need to take the average over batch axis
        np.mean(upstream_gradient, axis=0, out=self.gradient_bias)

        # Return downstream gradient
        return np.dot(upstream_gradient, np.transpose(self.weights), out=self.downstream)

    def model_setup(self, batch_size: int, in_shape: int, rho: float) -> Self:
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
            raise ValueError('The value of sigma must be greater than 0')
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

    def model_setup(self, batch_size, in_shape, **kwargs) -> Self:
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

    def __init__(self, batch_size: int, input_shape: int, rho: float):
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

    def add(self, layer: Linear | HyperTangent) -> Self:
        """
        Method adding a layer to the model. The method also triggers layer initialization.
        :param layer: The layer object.
        :return: The object itself.
        """
        # Trigger initialization with batch size and input shape
        self.layers.append(layer.model_setup(batch_size=self.batch_size,
                                             in_shape=self.current_out_shape,
                                             rho=self.rho))
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
            upstream_gradient = layer.backprop(upstream_gradient)  # Get upstream gradient for following layer
            if isinstance(layer, Linear):  # If layer is linear, layer.backprop also computes local gradient
                local_gradient = layer.gradient   # Pointer to current local gradient
                gradient_bias = layer.gradient_bias  # Pointers to current bias gradient
                gradient_list.extend([local_gradient, gradient_bias])   # Extend gradient list with two pointers

        # Flatten everything in C-contigous ordering
        gradient_list = [np.reshape(x, -1) for x in reversed(gradient_list)]
        gradient = np.concatenate(gradient_list)  # Concatenate the gradient vectors in one long gradient vector

        return gradient

    def evaluate_loss(self, train_data: np.ndarray, labels: np.ndarray, current_params: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Method evaluating the L2-penalized loss for the training data.
        It returns the value of the loss and the gradient.
        :param train_data: The training data, as a NumPy array.
        :param labels: The response data, as a 1-D NumPy array.
        :param current_params: The **ordered** parameters of the network in a 1-D NumPy array.
        :return: A tuple with the value of the loss and the gradient vector.
        """

        if self.layers[-1].out.shape[-1] != 1:
            raise ValueError('The last layer needs to have just one neuron, since it is the output one '
                             'and the output is scalar valued')

        # Setting the parameters in the net
        pos = 0  # Variable storing the slice position
        for layer in self.layers:
            if isinstance(layer, Linear):  # We have parameters only for the Linear layer
                if len(current_params) < pos + 1:  # Check that number of passed parameters is enough
                    raise ValueError('The length of the current_params array is not enough to cover '
                                     'the number of parameters')
                slice_dim_bias = layer.bias.shape[0]  # Get bias dimension
                slice_dim_weights = math.prod(layer.weights.shape)  # Get number of weights in weight matrix
                layer.bias[:] = current_params[pos: pos+slice_dim_bias]  # Set params in pre-allocated bias array
                pos += slice_dim_bias  # Update slice position w.r.t. input 1-D params array
                layer.weights.flat[:] = current_params[pos: pos+slice_dim_weights]  # Set params in weights array
                pos += slice_dim_weights  # Update slice position w.r.t. input 1-D params array

        out = train_data  # Initialize layer input
        reg = 0  # Initialize L2 penalty
        for layer in self.layers:  # Forward pass, layer per layer
            out = layer(out)  # Computing this pass we are also storing info necessary for the backward pass
            if isinstance(layer, Linear):  # If layer is linear, we add to the penalty the norm of the weight matrix
                reg += np.linalg.norm(layer.weights)

        out = np.squeeze(out)  # Squeeze the final output to avoid problems with broadcasting
        out = 1 / (1 + np.exp(out))  # Apply sigmoid activation
        # Cross entropy loss + regularization
        cross_entropy = -np.mean(labels * np.log(out) + (1 - labels) * np.log(1 - out)) + self.rho * reg

        downstream_grad = -labels / np.squeeze(out) + (1 - labels) / (1 - np.squeeze(out))  # Cross-entropy gradient
        # Downstream grad for the rest of the backprop, exploiting the nice analytical shape of the sigmoid derivative
        downstream_grad = (out * (1-out))*downstream_grad
        gradient = self.backprop(downstream_grad[:, np.newaxis])  # Start the backpropagation pipeline

        return cross_entropy, gradient

    def evaluate(self, test_data: np.ndarray) -> np.ndarray:
        """
        Method returning a 1-D array of predictions.
        :params test_data: The array (compatible with the initialized and trained model) containing the test data
        :returns: The 1-D array of predictions
        """
        pass


if __name__ == '__main__':
    from data_import import csv_import
    generator = np.random.default_rng(1234)
    labels, train_data = csv_import(['S', 'M'], '../data.txt', dtype=np.float64)
    model = Model(64, 16, 0)
    model.add(HyperTangent(0.5))
    model.add(Linear(1))
    model.evaluate_loss(train_data[:64, :-1], train_data[:64, -1],
                        current_params=np.concatenate([np.array([0.1, 0.1+1e-5]), generator.normal(size=15)]))
