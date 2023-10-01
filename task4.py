import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Function to return collected data, and returning it after splitting it.
def prepare_data(dataset_path):  
    data = pd.read_csv(dataset_path)  # Load your skin cancer csv file
    X = data.loc[:, "pixel0000":"pixel2351"]  # Select columns from "pixel000" to "pixelNNN" (replace NNN with the actual column name)
    y = data["label"]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # splitting the data
    return X_train, X_test, y_train, y_test


# Setting up the MLP Architecture
def create_mlp(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.0001, max_iter=1000): # setting up the parameters
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter) # creating the object
    return mlp

# Training Data by using model.fit
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    
# Displaying the Training Error
def plot_training_error(model, X_train, y_train):
    plt.plot(model.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Error")
    plt.show()

# Appling Trained Model on Test Data
def test_model(model, X_test):
    y_pred = model.predict(X_test) #receiving the predicted 
    return y_pred

# Display Results as a Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels) # creating a confusion matrix object 
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) #setting all of the parameters to enhance visualization
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Calculating Accuracy, Precision, Recall, and F-Measure
def evaluate_metrics(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred) # obtaining accuracy score
    precision = precision_score(y_true, y_pred, average='weighted', labels=labels) # obtaining precision score
    recall = recall_score(y_true, y_pred, average='weighted', labels=labels) # obtaining recall score
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels) # obtaining f1 score
    return accuracy, precision, recall, f1        

# Main function to run the pipeline
def main(dataset_path):
    X_train, X_test, y_train, y_test = prepare_data(dataset_path) # obtaining split data from directory/path
    
    # Defining MLP Architecture
    mlp = create_mlp(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.1, max_iter=300)
    
    # Fitting the Training Data
    train_model(mlp, X_train, y_train)
    
    # Displaying Training Error
    plot_training_error(mlp, X_train, y_train)
    
    # Applying Trained Model on Test Data
    y_pred = test_model(mlp, X_test)
    
    # Displaying Results as a Confusion Matrix
    labels = [0, 1, 2, 3, 4, 5, 6]  # Replace with your class labels
    plot_confusion_matrix(y_test, y_pred, labels)
    
    # Calculating Accuracy, Precision, Recall, and F-Measure
    accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred, labels)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Example usage
if __name__ == '__main__':
    dataset_path = './archive/hmnist_28_28_RGB.csv'
    main(dataset_path)
