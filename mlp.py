import numpy as np

class MLPClassifier:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            weight_shape = (layer_sizes[i], layer_sizes[i+1])
            bias_shape = (1, layer_sizes[i+1])
            self.weights.append(np.random.randn(*weight_shape))
            self.biases.append(np.random.randn(*bias_shape))

    def forward_propagation(self, X):
        activations = [X]
        for i in range(len(self.weights)):
            weighted_sum = np.dot(activations[i], self.weights[i]) + self.biases[i]
            activation = self.tanh(weighted_sum)
            activations.append(activation)
        return activations


    def backward_propagation(self, X, y, activations, learning_rate):
        deltas = [(activations[-1] - y) * self.tanh_derivative(activations[-1])]
        
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.tanh_derivative(activations[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0)

    def fit(self, X_train, y_train, learning_rate=0.1, epochs=1000):
        for epoch in range(epochs):
            #X_train, y_train = shuffle(X_train, y_train)
            activations = self.forward_propagation(X_train)
            self.backward_propagation(X_train, y_train, activations, learning_rate)

    def predict(self, X):
        activations = self.forward_propagation(X)
        predictions = np.argmax(activations[-1], axis=1)
        return predictions
    
    def test(self, X_test, y_test):
        prediction = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, prediction)
        
        return accuracy 
        

    def tune(self, X_train, y_train, param_grid, cv=5):
        # Perform grid search to find the best hyperparameters
        best_accuracy = 0
        best_params = {}

        for hidden_layers in param_grid['hidden_layers']:
            for learning_rate in param_grid['learning_rate']:
                for epochs in param_grid['epochs']:
                    mlp = MLPClassifier(input_size=self.input_size, hidden_layers=hidden_layers,output_size=self.output_size)
                    mlp.fit(X_train, y_train, learning_rate=learning_rate, epochs=epochs)
                    accuracy = mlp.test2(X_train, y_train)
                    print(accuracy)
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            'hidden_layers': hidden_layers,
                            'learning_rate': learning_rate,
                            'epochs': epochs
                        }

        self.hidden_layers = best_params['hidden_layers']
        self.learning_rate = best_params['learning_rate']
        self.epochs = best_params['epochs']

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