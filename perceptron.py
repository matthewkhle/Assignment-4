# -------------------------------------------------------------------------
# AUTHOR: Matthew Le
# FILENAME: perceptron.py
# SPECIFICATION: The program splits the data to x and y training arrays and the perceptron predicts with the test data. The highest accuracy is recorded and printed each time there is a new highest.
# FOR: CS 4200- Assignment #4
# TIME SPENT: 30 Minutes (an extra 60 minutes to solve import error on Vs Code)
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.linear_model import Perceptron
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tra', sep=',', header=None)

# getting the first 64 fields to form the feature data for training
X_training = np.array(df.values)[:, :64]
# getting the last field to form the class label for training
y_training = np.array(df.values)[:, -1]

# reading the data by using Pandas library
df = pd.read_csv('optdigits.tes', sep=',', header=None)

# getting the first 64 fields to form the feature data for test
X_test = np.array(df.values)[:, :64]
# getting the last field to form the class label for test
y_test = np.array(df.values)[:, -1]

highestAccuracy = 0

for a in n:  # iterates over n

    for b in r:  # iterates over r

        # Create the perceptron classifier
        # eta0 = learning rate, random_state = used to shuffle the training data
        clf = Perceptron(eta0=a, random_state=b, max_iter=1000)

        # Fitperceptron to the training data
        clf.fit(X_training, y_training)

        # make the classifier prediction for each test sample and start computing its accuracy
        # hint: to iterate over two collections simultaneously with zip() Example:
        # for (x_testSample, y_testSample) in zip(X_test, y_test):
        # to make a prediction do: clf.predict([x_testSample])
        # --> add your Python code here
        total = 0
        correct = 0
        for (x_testSample, y_testSample) in zip(X_test, y_test):
            class_prediction = clf.predict([x_testSample])
            if (class_prediction == y_testSample):
                correct += 1
            total += 1
        accuracy = correct / total
        # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together with the perceprton hyperparameters
        #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=00.1, random_state=True"
        # --> add your Python code here
        if (accuracy > highestAccuracy):
            highestAccuracy = accuracy
            print("Highest Perception accuracy so far: " + str(highestAccuracy) +
                  " Parameters: learning rate=" + str(a) + ", random_state=" + str(b))
