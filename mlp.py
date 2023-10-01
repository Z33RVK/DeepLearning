import numpy as np

class MLPClassifier:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size #the input dimension , basically the number of rows
        self.hidden_layers = hidden_layers #number of layers
        self.output_size = output_size #the number of output classes
        self.weights = [] # the list of weights
        self.biases = [] # the list of biases

        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_layers + [output_size] # summation of all neurons
        for i in range(len(layer_sizes) - 1):
            weight_shape = (layer_sizes[i], layer_sizes[i+1]) # setting the weight shape based on the layer size
            bias_shape = (1, layer_sizes[i+1]) # setting the bias shape based on the layer size
            self.weights.append(np.random.randn(*weight_shape)) # appending the weights through random function of np from weight list
            self.biases.append(np.random.randn(*bias_shape)) # appending the biases through random function of np from bias list

    def forward_propagation(self, X):
        activations = [X] # list of activations
        for i in range(len(self.weights)): # traversing all weights in list
            weighted_sum = np.dot(activations[i], self.weights[i]) + self.biases[i] # dot product and bias calculated
            activation = self.tanh(weighted_sum) # passed through activation function
            activations.append(activation) # result appended to function list
        return activations


    def backward_propagation(self, X, y, activations, learning_rate):
        deltas = [(activations[-1] - y) * self.tanh_derivative(activations[-1])] # error calculated and multipled with derivative calculated of same neuron in back pass
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.tanh_derivative(activations[i]) # error calculated and multipled with derivative calculated of same neuron in back pass
            deltas.insert(0, delta) # error inserted into delta list

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i]) # learning multiplied with the dot product with activation result and error , subtracted from the original weight
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0) # learning multiplied with the dot product with activation result and error , subtracted from the original weight

    def fit(self, X_train, y_train, learning_rate=0.1, epochs=1000): # model fitting  
        for epoch in range(epochs):
            #X_train, y_train = shuffle(X_train, y_train)
            activations = self.forward_propagation(X_train) # calling forward proagation
            self.backward_propagation(X_train, y_train, activations, learning_rate) # calling backward propagation

    def predict(self, X): # function to predict
        activations = self.forward_propagation(X) # activations are obtained
        predictions = np.argmax(activations[-1], axis=1)  # converts a matrix of predicted probabilities into an array of predicted class labels
        return predictions
    
    def test(self, X_test, y_test):
        prediction = self.predict(X_test) # predicting on the test set
        
        accuracy = accuracy_score(y_test, prediction) # obtaining the accuracy
        
        return accuracy 
        
    #below are all of the activation function    
    def sigmoid(self, x):
        # Clip the values to prevent overflow or underflow
        clipped_x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-clipped_x))

    def sigmoid_derivative(self, x):
            return x * (1 - x)
    
    def relu(self,x):
        return np.maximum(0, x)

    def relu_derivative(self,x):
        return np.where(x > 0, 1, 0)    
    
    def leaky_relu(self, x, alpha=0.01):
        return np.where(x >= 0, x, alpha * x)

    def leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x >= 0, 1, alpha)

    def tanh(self, x):
        return np.tanh(x)    

    def tanh_derivative(self,x):
        return 1 - np.tanh(x) ** 2