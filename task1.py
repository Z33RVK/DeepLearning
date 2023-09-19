from sklearn.datasets import load_files

#path to dataset
yale_dataset_path = "./archive" 

# loading the Yale image dataset
yale_dataset = load_files(yale_dataset_path, shuffle=False)

# Access the data and target labels
X = yale_dataset.data  # Data (images)
Y = yale_dataset.target  # Target labels

##print(X)
##print(Y)

#Splitting the dataset into testing and training sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


