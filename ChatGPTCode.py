# This is a sample backpropagation script.

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)  # sigmoid derivative is  x*(1-x) if x is the layer activation, and it is sigmoid(x) * (1-sigmoid(x)) if x is the layer pre-activation


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_weights = np.random.rand(self.input_size, self.hidden_size)
        self.hidden_bias = np.random.rand(1, self.hidden_size)
        self.output_weights = np.random.rand(self.hidden_size, self.output_size)
        self.output_bias = np.random.rand(1, self.output_size)

    def forward(self, inputs):
        self.hidden_activation = sigmoid(np.dot(inputs, self.hidden_weights) + self.hidden_bias)
        self.output = sigmoid(np.dot(self.hidden_activation,self.output_weights) + self.output_bias) # shape of self.output is (4,1)

    def backward(self, inputs, targets, learning_rate):
        """
            backpropagation is performed over the entire dataset at once

            derivative of loss function w.r.t output layer weights and biases
            dl/dw_o = dl/dA_o * dA_o/dz_o * dz_o/dw_o = (t-A_o)*s_d(A_o)*A_h
            dl/db_o = dl/dA_o * dA_o/dz_o * dz_o/db_o = (t-A_o)*s_d(A_o)

            In this implementation, terms regrouped and renamed, so the backpropagation is more readable
            output_error = dl/dA_o
            output_delta = output_error * dA_o/dz_o

            so, dl/dw_o = l_r * A_h * output_delta.T and
            dl/db_o = l_r * ∑(output_delta_ij) along axis 0

            ------------------------------------------------------
            derivative of loss function w.r.t hidden layer weights and biases
            dl/dw_h =  output_delta * dz_o/dA_h * dA_h/dz_h * dz_h/dw_h
            dl/db_h = output_delta * dz_o/dA_h * dA_h/dz_h * dz_h/db_h

            hidden_error = output_delta * dz_o/dA_h
            hidden_delta = hidden_error * dA_h/dz_h

            so, dl/dw_h = l_r * x.T * hidden_delta,  and
            dl/db_h = l_r * ∑(hidden_delta_ij) along axis 0

        """

        output_error = targets - self.output # shape = (4,1)
        output_delta = output_error * sigmoid_derivative(self.output) # element_wise multiplication, output_delta shape(4,1)
        hidden_error = np.dot(output_delta,self.output_weights.T) # (4,2)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_activation) # element-wise multiplication (4,2)

        self.output_weights += learning_rate * np.dot(self.hidden_activation.T, output_delta )  # inner product (2,1)
        self.output_bias += learning_rate * np.sum(output_delta, axis=0, keepdims=True)

        self.hidden_weights += learning_rate * np.dot(inputs.T, hidden_delta) # shape = ((2,4) * (4,2) = (2,2))
        self.hidden_bias += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) # keepdims parameter ensures that the output retains the number of dimensions as the input

    def train(self, inputs, targets, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(inputs)
            self.backward(inputs, targets, learning_rate)
            loss = np.mean(np.square(targets - self.output))
            print("Epoch {}/{}, Loss {:.4f}".format((epoch + 1), epochs, loss))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.random.seed(2)
    NN = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

    # input and output for XOR problem
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])

    # training using batch gradient descent
    NN.train(inputs, targets, epochs=10000, learning_rate=0.1)
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # evaluate the model
    NN.forward(test_inputs)
    print("predictions:", NN.output)



