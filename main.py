import numpy as np

# Set random seed for reproducibility
np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_input_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden_output = np.zeros((1, output_size))

    def feedforward(self, inputs):
        self.hidden_activation = sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_input_hidden)
        self.output = sigmoid(np.dot(self.hidden_activation, self.weights_hidden_output) + self.bias_hidden_output)
        return self.output

    def backward(self, inputs, targets, learning_rate):
        error = targets - self.output
        output_delta = error * self.output * (1 - self.output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.hidden_activation * (1 - self.hidden_activation)
        
        self.weights_hidden_output += np.dot(self.hidden_activation.T, output_delta) * learning_rate
        self.bias_hidden_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(inputs.T, hidden_delta) * learning_rate
        self.bias_input_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

input_size = 10
hidden_size = 64
output_size = 1
learning_rate = 0.1
nn = NeuralNetwork(input_size, hidden_size, output_size)

data = np.random.random((1000, input_size))
labels = np.random.randint(2, size=(1000, output_size))

for epoch in range(10):
    output = nn.feedforward(data)
    nn.backward(data, labels, learning_rate)

# Print last calculated values
print("Last output:")
print(output)
