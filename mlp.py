import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, hidden_layers=(100,), activation='relu', learning_rate=0.01, momentum=0.9, epochs=100):
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.weights = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    
    def initialize_weights(self, input_dim, output_dim):
        self.weights.append(np.random.randn(input_dim, self.hidden_layers[0]))
        for i in range(len(self.hidden_layers) - 1):
            self.weights.append(np.random.randn(self.hidden_layers[i], self.hidden_layers[i+1]))
        self.weights.append(np.random.randn(self.hidden_layers[-1], output_dim))
    
    def forward_propagation(self, X):
        self.layers = []
        input_data = X
        for i in range(len(self.weights)):
            self.layers.append(input_data)
            net_input = np.dot(input_data, self.weights[i])
            if self.activation == 'sigmoid':
                input_data = self.sigmoid(net_input)
            elif self.activation == 'relu':
                input_data = self.relu(net_input)
        self.layers.append(input_data)
    
    def backward_propagation(self, X, y):
        error = y - self.layers[-1]
        deltas = [error * self.sigmoid_derivative(self.layers[-1])]
        
        for i in range(len(self.layers) - 2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            if self.activation == 'sigmoid':
                delta = error * self.sigmoid_derivative(self.layers[i])
            elif self.activation == 'relu':
                delta = error * self.relu_derivative(self.layers[i])
            deltas.append(delta)
        
        for i in range(len(self.weights) - 1, -1, -1):
            self.weights[i] += self.learning_rate * self.layers[i].T.dot(deltas.pop())
    
    def fit(self, X, y):
        self.initialize_weights(X.shape[1], y.shape[1])
        self.train_accuracies = []
        self.valid_accuracies = []
        
        for epoch in range(self.epochs):
            self.forward_propagation(X)
            self.backward_propagation(X, y)
            
            train_pred = self.predict(X)
            train_acc = accuracy_score(np.argmax(y, axis=1), np.argmax(train_pred, axis=1))
            self.train_accuracies.append(train_acc)
            
            valid_pred = self.predict(X_valid)
            valid_acc = accuracy_score(np.argmax(y_valid, axis=1), np.argmax(valid_pred, axis=1))
            self.valid_accuracies.append(valid_acc)
            
            print(f"Epoch {epoch+1}/{self.epochs} - Train Accuracy: {train_acc:.4f} - Validation Accuracy: {valid_acc:.4f}")
        
    def predict(self, X):
        self.forward_propagation(X)
        return self.layers[-1]
    
    def test(self, X, y):
        y_pred = self.predict(X)
        acc = accuracy_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        precision = precision_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        recall = recall_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        f1 = f1_score(np.argmax(y, axis=1), np.argmax(y_pred, axis=1), average='weighted')
        confusion = confusion_matrix(np.argmax(y, axis=1), np.argmax(y_pred, axis=1))
        
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion)
    
    def tune(self, X, y, param_grid):
        model = GridSearchCV(estimator=self, param_grid=param_grid, cv=5, scoring='accuracy')