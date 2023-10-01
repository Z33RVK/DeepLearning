from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
from mlp import MLPClassifier
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def analyze_input_data(X):
    data_format = None
    data_type = None
    
    if isinstance(X, np.ndarray):
        data_format = "numpy array"
        data_type = X.dtype
    elif isinstance(X, pd.DataFrame):
        data_format = "Pandas DataFrame"
        data_type = X.dtypes
    elif isinstance(X, list):
        data_format = "list"
        data_type = type(X[0])
    elif isinstance(X, str):
        data_format = "string"
        data_type = "Not applicable"
    else:
        data_format = "unknown"
        data_type = "unknown"
    
    return data_format, data_type

 #path to dataset

# Define a function to extract the person label from the file name
def extract_person_label(file_name):
    return int(file_name.split('.')[0].replace('subject', '')) - 1  # Subtract 1 to make labels start from 0

yale_dataset_path = "./archive" 

# # loading the Yale image dataset
yale_dataset = load_files(yale_dataset_path, shuffle=False)

data = []  # List to store image data
labels = []  # List to store labels

for file_name in os.listdir(yale_dataset_path):
    file_path = os.path.join(yale_dataset_path, file_name)
    
    if os.path.isdir(file_path):
        continue  # Skip directories
    
    if file_name == 'Readme.txt':
        continue  # Skip the 'Readme.txt' file

    img = plt.imread(file_path)
    #print(img.shape)
    data.append(img.flatten())  # Flatten image into a 1D array
    labels.append(extract_person_label(file_name))

data = np.array(data)
labels = np.array(labels)

print("Dataset: ")
print(data.shape)
print("Labels: ")
print(labels.shape)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Fit the LabelEncoder on the training labels
label_encoder.fit(y_train)

# Convert the training and test labels to numerical format
y_train_encoded = label_encoder.transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

mlp = MLPClassifier(input_size=77760, hidden_layers=[64, 32], output_size=15)

# Convert the target labels to numerical format and reshape it
y_train_encoded = label_encoder.fit_transform(y_train)
y_train_encoded = y_train_encoded.reshape(-1, 1)

# Train the model
mlp.fit(X_train, y_train_encoded, learning_rate = 0.1, epochs = 1000)

# Obtain predictions
y_pred_encoded = mlp.predict(X_test)

# Obtain predictions
y_pred = mlp.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute precision
precision = precision_score(y_test, y_pred, average='micro')

# Compute recall
recall = recall_score(y_test, y_pred, average='micro')

# Compute F1-score
f1 = f1_score(y_test, y_pred, average='micro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# This implementation assumes that you have already loaded the Yale dataset and labels into the variables `dataset` and `labels`, respectively. You will need to adjust the file paths accordingly. The dataset should be in the shape `(num_samples, num_features)` and the labels should be in the shape `(num_samples,)`.

# The `MLPClassifier` class provides the necessary methods for training the model (`fit`), testing the model (`test`), and tuning hyperparameters (`tune`). The `fit` method trains the model using backpropagation, the `test` method evaluates the model's accuracy on a test set, and the `tune` method performs grid search to find the best hyperparameters.

# To use the `tune` method, you can specify a parameter grid containing different values for the hyperparameters you want to tune. For example:

param_grid = {
    'hidden_layers': [[32], [64, 32], [128, 64, 32]],
    'learning_rate': [0.1, 0.01, 0.001],
    'epochs': [100, 500, 1000]
}

# Create an instance of LabelBinarizer
label_binarizer = LabelBinarizer()

# Fit and transform the training labels to binary format
y_train_bin = label_binarizer.fit_transform(y_train)

mlp.tune(X_train, y_train_bin, param_grid)