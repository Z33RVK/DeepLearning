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

yale_dataset_path = "./archive1" 

# # loading the Yale image dataset
yale_dataset = load_files(yale_dataset_path, shuffle=False)

data = []  # List to store image data
labels = []  # List to store labels

for file_name in os.listdir(yale_dataset_path):
    try:
        img = plt.imread(os.path.join(yale_dataset_path, file_name))
    except (IOError, OSError):
        continue

    #print(img)
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
output_labels = len(np.unique(labels))

y_train = y_train.reshape(-1,1)

mlp = MLPClassifier(input_size=X_train.shape[1], hidden_layers=[128, 64], output_size=output_labels)

# Train the model
mlp.fit(X_train, y_train, learning_rate = 0.1, epochs = 20)

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
