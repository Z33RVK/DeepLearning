import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Task 1: Data Preparation
def prepare_data(dataset_path):
    data = pd.read_csv(dataset_path)  # Load your skin cancer dataset
    X = data.loc[:, "pixel0000":"pixel2351"]  # Select columns from "pixel000" to "pixelNNN" (replace NNN with the actual column name)
    y = data["label"]  # Labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Task 2: Display N Training or Test Samples
def display_samples(X, y, n_samples):
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    for i, index in enumerate(sample_indices):
        plt.subplot(1, n_samples, i + 1)
        # Assuming that your data represents images of shape (height, width), you should reshape it accordingly
        # Replace (64, 64) with the actual dimensions of your images.
        image_data = X.iloc[index].values.reshape((64, 64))
        plt.imshow(image_data, cmap='gray')
        plt.title(f"Class {y.iloc[index]}")
        plt.axis('off')
    plt.show()

# Task 3: Define MLP Architecture
def create_mlp(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.0001, max_iter=1000):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, alpha=alpha, max_iter=max_iter)
    return mlp

# Task 4: Fit the Training Data
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    
# Task 5: Display Training Error
def plot_training_error(model, X_train, y_train):
    plt.plot(model.loss_curve_)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Error")
    plt.show()

# Task 6: Apply Trained Model on Test Data
def test_model(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

# Task 7: Display Results as a Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Task 8: Calculate Accuracy, Precision, Recall, and F-Measure
def evaluate_metrics(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', labels=labels)
    recall = recall_score(y_true, y_pred, average='weighted', labels=labels)
    f1 = f1_score(y_true, y_pred, average='weighted', labels=labels)
    return accuracy, precision, recall, f1        

# Main function to run the pipeline
def main(dataset_path):
    X_train, X_test, y_train, y_test = prepare_data(dataset_path)
    
    # Task 2: Display N Training or Test Samples
    # display_samples(X_train, y_train, 5)
    
    # Task 3: Define MLP Architecture
    mlp = create_mlp(hidden_layer_sizes=(100, 100), activation='relu', alpha=0.1, max_iter=300)
    
    # Task 4: Fit the Training Data
    train_model(mlp, X_train, y_train)
    
    # Task 5: Display Training Error
    plot_training_error(mlp, X_train, y_train)
    
    # Task 6: Apply Trained Model on Test Data
    y_pred = test_model(mlp, X_test)
    
    # Task 7: Display Results as a Confusion Matrix
    labels = [0, 1, 2, 3, 4, 5, 6]  # Replace with your class labels
    plot_confusion_matrix(y_test, y_pred, labels)
    
    # Task 8: Calculate Accuracy, Precision, Recall, and F-Measure
    accuracy, precision, recall, f1 = evaluate_metrics(y_test, y_pred, labels)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

# Example usage
if __name__ == '__main__':
    dataset_path = './archive/hmnist_28_28_RGB.csv'
    main(dataset_path)
