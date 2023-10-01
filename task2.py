import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf 
from sklearn.datasets import load_files 
import os
#Part 2

# Taken from sir's notes
# Define a function to extract the person label from the file name 
def extract_person_label(file_name):
    return int(file_name.split('.')[0].replace('subject', '')) - 1  # Subtract 1 to make labels start from 0

yale_dataset_path = "./archive" 

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

#print(data.shape) -> (165, 77760)
#print(labels.shape) -> (165)
#print(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# print(X_train.shape) # (132, 77760)
# print(X_test.shape) # (33, 77760)
# print(y_train.shape) # (132, )
# print(y_test.shape) # (33, )

# input size
X_train_columns = len(X_train[0]) #77760

#number of unique labels
output_labels = len(np.unique(y_train))
print("Output labels: ",output_labels)

# Define your MLP architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(output_labels, activation='softmax')
])

# Compile the model with categorical cross-entropy loss
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=30)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

#calculating the precision, recall and f1-score
precision = precision_score(y_test, y_pred_classes, average="micro")
recall = recall_score(y_test, y_pred_classes, average="micro")
f1 = f1_score(y_test, y_pred_classes, average="macro")

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

