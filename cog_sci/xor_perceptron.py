import numpy as np

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def sigmoid_derivative(f):
    return f * (1 - f)

def initialize_parameters():
    np.random.seed(42)
    weights_input_hidden = np.array([[6.0, 8.0], [-6.0, -8.0]])
    weights_hidden_output = np.array([[6.0], [-6.0]])
    bias_hidden = np.array([[ 2.0,  -1.0 ]])
    bias_output = np.array([[ -2.0 ]])

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    hidden_layer_input = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output_layer_output = sigmoid(output_layer_input)

    return hidden_layer_output, output_layer_output

def backward_propagation(inputs, outputs, targets, hidden_layer_output, output_layer_output,
                         weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate):
    output_error = targets - output_layer_output
    output_delta = output_error * sigmoid_derivative(output_layer_output)

    hidden_error = output_delta.dot(weights_hidden_output.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

def train_neural_network(inputs, targets, epochs, learning_rate):

    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters()

    for epoch in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation( 
                inputs, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

        backward_propagation(inputs, output_layer_output, targets, hidden_layer_output, output_layer_output,
                             weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate)

        loss = np.mean(np.square(targets - output_layer_output))
        print(f'Epoch {epoch}, Loss: {loss}')

    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Training data : inputs and targets
inputs  = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

trained_weights_input_hidden, trained_weights_hidden_output, trained_bias_hidden, trained_bias_output = train_neural_network(
    inputs, targets, epochs=2, learning_rate=0.1)

# Test the trained model
test_input = np.array([[0, 1]])
hidden_output, output = forward_propagation(
    test_input, trained_weights_input_hidden, trained_weights_hidden_output, trained_bias_hidden, trained_bias_output)

print("Test Output:", output)

