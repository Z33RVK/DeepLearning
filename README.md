# DeepLearning

## Task - 1

### Consider the Yale dataset that I also used in the class. The database contains 165 GIF images of 15 subjects (subject01, subject02, etc.). I already provided a boilerplate code, which you can use for this assignment.

### Link: https://www.kaggle.com/datasets/olgabelitskaya/yale-face-database

- Implement your own code for MLP architecture. Preferably, create a class for MLP and implement the necessary methods for the model. Three core functions that you must implement are: fit() for training the model, test() for testing the model and tune() for tuning the hyperparameters of the model. The model must have the ability to create multiple hidden layers and varying numbers of neurons as set by the user. Apply proper implementation to update the weights using Backpropagation.
- You can use Scikitlearn’s GridSearchCV for hyperparameter turning. At least the following parameters must be fine-tuned: Number of Neurons per Layer, Number of Hidden Layers, Activation Functions, Epochs, Learning Rate, and Momentum.
- For evaluation, you must use, accuracy, precision, recall, and f1-score.
- You must run two sets of experiments, one with parameter tuning and one without
parameter tuning. For each experiment, you must create tables to show the results. Also,
display the confusion metrics.
- For each epoch, plot the train and validation accuracies.

## Task - 2

You can implement the model for the above dataset using Scikit Learn or any available library. Show the results, using the same metrics as above. Compare this result with your custom implementation done as done in Task 1.

## Task - 3

In this task, you will be tasked with developing face recognition software for employees of FAST (assuming all face pictures provided in Yale DB belong to FAST employees). For each image available in Yale database, you need to crop only the face region by using Haar cascade classifier available in OpenCV. Resize the image to a fixed size and then save them in a
 
folder. Finally, train the model on these cropped images. However, the system should be designed to accommodate new employees joining the organization. When a new employee joins, their face picture will be captured, and the existing model needs to be updated incrementally, rather than training it from scratch. The following steps need to be completed:
1. Model Training: Develop the code to train the face recognition model in an incremental fashion. This means that when a new employee's face picture is added, the model should update itself without losing its previous knowledge of recognizing existing employees.
2. Deployment: Deploy the trained model on a local host or a freely available host, which can be found at https://wiki.python.org/moin/FreeHosts.
This deployment should expose the model through API calls, enabling it to be accessed from various client applications such as web or mobile apps.
3. Web Interface for Registration: Create a user-friendly web interface where individuals can register by providing their name, email, and capturing a series of pictures (two or more). Generate a unique ID for each user and store this information in an SQLite database. Additionally, train the face recognition model to include the new user, allowing the system to recognize their face.
4. Face Recognition Page: Develop another web page where a registered user can use their camera to show their face. The system should accurately identify the user based on their facial features. If the recognition is successful, display the user's name; otherwise, show a message indicating "Unknown user."

## Task - 4

You need to classify the skin cancer data into 7 different classes, one class for each type of skin cancer. You need to implement a multi-layer perceptron (MLP) architecture. The overall accuracy should reach at least 80%.
For each of the following tasks, create a separate function (where possible)
• Read all data and divide into training and testing sets (usually 80% for training and 20% for testing)
• Define a function to show N training or test samples, N is the parameter for the function.
 
• Define a function to create an MLP architecture with various parameters and fit the training data
• Display the training error
• Apply the trained model on test data
• Display the results as a confusion matrix
• Find accuracy on test data using Precision, Recall, and f-Measure
• Write a report like an “Experimental Results” section in a research paper by describing
the technical details of your experiments. (Refer to the sample article I attached with this assignment and specifically read the “Experiment and Evaluation” section to write your report.

