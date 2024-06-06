import numpy as np           # Import the numpy library for numerical operations
import pandas as pd          # Import the pandas library for data manipulation
import csv                   # Import the csv module for reading CSV files
import random                # Import the random module for generating random numbers
import math                  # Import the math module for mathematical operations
from sklearn.model_selection import cross_val_score, train_test_split  # Import functions from scikit-learn for model selection
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier from scikit-learn for KNN classification

# Load the dataset from a CSV file
def loadd(f):               # Define a function 'loadd' that takes a filename 'f' as an argument
    d = []                  # Create an empty list 'd' to store the dataset
    with open(f, 'r') as f:  # Open the file 'f' in read mode using a context manager
        c_read = csv.reader(f)  # Create a CSV reader for the file
        for r in c_read:    # Iterate through the rows in the CSV file
            d.append(r)     # Append each row to the dataset list
    return d               # Return the dataset

def dataset(f):            # Define a function 'dataset' that takes a filename 'f' as an argument
    return loadd(f)        # Call the 'loadd' function to load and return the dataset

# Convert dataset values to numeric (if needed)
def convert_n(d):          # Define a function 'convert_n' that takes a dataset 'd' as an argument
    for i in range(len(d[0])):  # Iterate through the columns in the dataset
        a = [row[i] for row in d]  # Extract the values in the column 'i'
        b = list(set(a))    # Create a list of unique values in the column
        for j in range(len(d)):  # Iterate through the rows in the dataset
            d[j][i] = b.index(d[j][i])  # Replace each value with its index in the unique values list
    return d               # Return the modified dataset

def convert2numeric(d):    # Define a function 'convert2numeric' that takes a dataset 'd' as an argument
    return convert_n(d)   # Call the 'convert_n' function to convert and return the dataset to numeric

# Split dataset into k folds
def valid(d, f):           # Define a function 'valid' that takes a dataset 'd' and a number of folds 'f' as arguments
    f_size = len(d) // f   # Calculate the size of each fold
    d_copy = list(d)       # Create a copy of the dataset
    d_folds = []           # Create an empty list to store the folds
    for _ in range(f):     # Repeat 'f' times to create 'f' folds
        folds = []         # Create an empty list for a fold
        while len(folds) < f_size:  # Repeat until the fold size is reached
            i = random.randrange(len(d_copy))  # Generate a random index
            folds.append(d_copy.pop(i))         # Add a data point to the fold and remove it from the copy
        d_folds.append(folds)  # Add the fold to the list of folds
    return d_folds          # Return the list of folds

def val_cross(d, f=10):    # Define a function 'val_cross' that takes a dataset 'd' and a number of folds 'f' as arguments
    return valid(d, f)     # Call the 'valid' function to split the dataset into folds and return them

# Calculate Euclidean distance between two data points
def euclid(p_1, p_2):      # Define a function 'euclid' that takes two data points 'p_1' and 'p_2' as arguments
    d = 0                 # Initialize a variable 'd' to store the Euclidean distance
    for i in range(len(p_1) - 1):  # Iterate through the features of the data points
        d += (p_1[i] - p_2[i]) ** 2  # Calculate the squared difference for each feature and add to 'd'
    return math.sqrt(d)     # Return the square root of 'd' as the Euclidean distance

def dist_euclid(p_1, p_2):  # Define a function 'dist_euclid' that takes two data points 'p_1' and 'p_2' as arguments
    return euclid(p_1, p_2)  # Call the 'euclid' function to calculate and return the Euclidean distance

# Get k nearest neighbors for a test instance
def neigh(train, testins, n):  # Define a function 'neigh' that takes a training dataset, a test instance, and the number of neighbors 'n' as arguments
    d = []                   # Create an empty list to store distances and corresponding training instances
    for trainins in train:    # Iterate through the training instances
        dist = dist_euclid(testins, trainins)  # Calculate the Euclidean distance between the test instance and the training instance
        d.append((trainins, dist))  # Append the training instance and its distance to the list
    d.sort(key=lambda l: l[1])    # Sort the list of distances in ascending order
    ne = [l[0] for l in d[:n]]    # Extract the 'n' nearest neighbors
    return ne                  # Return the 'n' nearest neighbors

def neighbors_g(train, testins, k):  # Define a function 'neighbors_g' that takes a training dataset, a test instance, and the number of neighbors 'k' as arguments
    return neigh(train, testins, k)  # Call the 'neigh' function to get the nearest neighbors and return them

# Make a prediction for a test instance
def prediction(train, testins, c):   # Define a function 'prediction' that takes a training dataset, a test instance, and the number of neighbors 'c' as arguments
    ne = neighbors_g(train, testins, c)  # Call the 'neighbors_g' function to get the nearest neighbors
    votes_class = {}              # Create a dictionary to store class votes
    for n in ne:                  # Iterate through the nearest neighbors
        r = n[-1]                 # Get the class label of the neighbor
        if r in votes_class:       # Check if the class label is in the dictionary
            votes_class[r] += 1    # Increment the vote count for the class
        else:
            votes_class[r] = 1     # Initialize the vote count for the class
    sort = sorted(votes_class.items(), key=lambda x: x[1], reverse=True)  # Sort the class votes in descending order
    return sort[0][0]             # Return the class label with the highest vote

def classpredict(train, testins, c):  # Define a function 'classpredict' that takes a training dataset, a test instance, and the number of neighbors 'c' as arguments
    return prediction(train, testins, c)  # Call the 'prediction' function to make a class prediction

# Evaluate the KNN algorithm using k-fold cross-validation
def getknn(d, k, f=10):         # Define a function 'getknn' that takes a dataset 'd', the number of neighbors 'k', and the number of folds 'f' as arguments
    c = []                     # Create an empty list to store accuracy scores
    fi = val_cross(d, f)       # Call the 'val_cross' function to split the dataset into folds
    for i in range(f):         # Iterate through the folds
        p = []                 # Create an empty list to store predictions
        train = list(fi)       # Create a copy of the folds
        train.pop(i)           # Remove the current fold from the training data
        train = sum(train, [])  # Flatten the list of training data
        test = fi[i]            # Get the test data from the current fold
        for testins in test:    # Iterate through the test instances in the current fold
            p_class = classpredict(train, testins, k)  # Make a class prediction for the test instance
            p.append(p_class)   # Append the prediction to the list of predictions
        a = [row[-1] for row in test]  # Extract the true class labels from the test data
        acc = calc_acc(a, p)    # Calculate accuracy for the fold
        c.append(acc)          # Append the accuracy score to the list
    return c                   # Return the list of accuracy scores

def knn(d, k, f=10):           # Define a function 'knn' that takes a dataset 'd', the number of neighbors 'k', and the number of folds 'f' as arguments
    return getknn(d, k, f)    # Call the 'getknn' function to evaluate the KNN algorithm and return the accuracy scores

# Calculate accuracy percentage
def acc(a, p):                 # Define a function 'acc' that takes true class labels 'a' and predicted class labels 'p' as arguments
    c = 0                      # Initialize a counter for correct predictions
    for i in range(len(a)):     # Iterate through the data points
        if a[i] == p[i]:        # Check if the true label matches the predicted label
            c += 1              # Increment the counter for correct predictions
    return (c / float(len(a))) * 100.0  # Calculate and return the accuracy percentage

def calc_acc(a, p):            # Define a function 'calc_acc' that takes true class labels 'a' and predicted class labels 'p' as arguments
    return acc(a, p)           # Call the 'acc' function to calculate and return the accuracy percentage

# Load and preprocess the Car Evaluation dataset
def preprocess_car():           # Define a function 'preprocess_car'
    c_d = dataset('car.data')    # Load the Car Evaluation dataset using the 'dataset' function
    c_d = convert2numeric(c_d)  # Convert the dataset to numeric
    return c_d                   # Return the preprocessed dataset

# Load and preprocess the Breast Cancer dataset
def preprocess_cancer():        # Define a function 'preprocess_cancer'
    ca_d = dataset('breast-cancer.data')  # Load the Breast Cancer dataset using the 'dataset' function
    ca_d = convert2numeric(ca_d)  # Convert the dataset to numeric
    return ca_d                  # Return the preprocessed dataset

# Load and preprocess the Hayes-Roth dataset
def preprocess_hr():             # Define a function 'preprocess_hr'
    hr_d = dataset('hayes-roth.data')  # Load the Hayes-Roth dataset using the 'dataset' function
    hr_d = convert2numeric(hr_d)  # Convert the dataset to numeric
    return hr_d                   # Return the preprocessed dataset

# Main function
if __name__ == '__main__':       # Check if the script is the main program
    random.seed(0.90)             # Set a random seed for reproducibility

    # Load and preprocess the datasets
    c_d = preprocess_car()        # Preprocess the Car Evaluation dataset
    ca_d = preprocess_cancer()    # Preprocess the Breast Cancer dataset
    hr_d = preprocess_hr()        # Preprocess the Hayes-Roth dataset

    # Specify the value of k for KNN
    k = 3

    # Evaluate the KNN algorithm using 10-fold cross-validation for your code from scratch
    knncar = knn(c_d, k, f=10)    # Evaluate KNN for the Car Evaluation dataset
    knncancer = knn(ca_d, k, f=10)  # Evaluate KNN for the Breast Cancer dataset
    knnhr = knn(hr_d, k, f=10)    # Evaluate KNN for the Hayes-Roth dataset

    # Print the results for your code from scratch
    print('Result from my implementation:')
    print('Evaluation - CAR:')
    print('Accuracy:', knncar)      # Print accuracy scores for the Car Evaluation dataset
    print('Mean Accuracy:', sum(knncar) / len(knncar))  # Calculate and print the mean accuracy

    print('\nEValuation - Breast Cancer')
    print('Accuracy:', knncancer)   # Print accuracy scores for the Breast Cancer dataset
    print('Mean Accuracy:', sum(knncancer) / len(knncancer))  # Calculate and print the mean accuracy

    print('\nEvaluation - Hayes Roth')
    print('Accuracy:', knnhr)       # Print accuracy scores for the Hayes-Roth dataset
    print('Mean Accuracy:', sum(knnhr) / len(knnhr))  # Calculate and print the mean accuracy

    # Load and preprocess the datasets for scikit-learn
    car = preprocess_car()          # Preprocess the Car Evaluation dataset
    cancer = preprocess_cancer()    # Preprocess the Breast Cancer dataset
    hr = preprocess_hr()            # Preprocess the Hayes-Roth dataset

    # Create K-NN classifier with scikit-learn
    class_knn = KNeighborsClassifier(n_neighbors=k)  # Create a K-NN classifier with the specified number of neighbors

    # Evaluate K-NN with 10-fold cross-validation for scikit-learn
    c_s = cross_val_score(class_knn, np.array(car)[:, :-1], np.array(car)[:, -1], cv=10) * 100.0  # Evaluate KNN for the Car Evaluation dataset with scikit-learn
    ca_s = cross_val_score(class_knn, np.array(cancer)[:, :-1], np.array(cancer)[:, -1], cv=10) * 100.0  # Evaluate KNN for the Breast Cancer dataset with scikit-learn
    hr_s = cross_val_score(class_knn, np.array(hr)[:, :-1], np.array(hr)[:, -1], cv=10) * 100.0  # Evaluate KNN for the Hayes-Roth dataset with scikit-learn

    # Print the results for scikit-learn
    print('\nResult from Scikit-learn:')
    print('Evaluation - Car:')
    print('Accuracy:', c_s)       # Print accuracy scores for the Car Evaluation dataset with scikit-learn
    print('Mean Accuracy:', np.mean(c_s))  # Calculate and print the mean accuracy

    print('\nEvaluation - Breast Cancer:')
    print('Accuracy scores:', ca_s)  # Print accuracy scores for the Breast Cancer dataset with scikit-learn
    print('Mean Accuracy:', np.mean(ca_s))  # Calculate and print the mean accuracy

    print('\nEvaluation - Hayes Roth:')
    print('Accuracy scores:', hr_s)  # Print accuracy scores for the Hayes-Roth dataset with scikit-learn
    print('Mean Accuracy:', np.mean(hr_s))  # Calculate and print the mean accuracy

    # Perform the paired t-test
    from scipy import stats

    # Calculate the differences in accuracy scores between your implementation and scikit-learn
    d_car = [a - b for a, b in zip(knncar, c_s)]  # Calculate the differences for the Car Evaluation dataset
    d_cancer = [a - b for a, b in zip(knncancer, ca_s)]  # Calculate the differences for the Breast Cancer dataset
    diff_hr = [a - b for a, b in zip(knnhr, hr_s)]  # Calculate the differences for the Hayes-Roth dataset

    # Calculate the mean and standard deviation of the differences
    mean_d_car = np.mean(d_car)  # Calculate the mean difference for the Car Evaluation dataset
    std_dev_car = np.std(d_car, ddof=1)  # Calculate the sample standard deviation for the Car Evaluation dataset

    mean_d_cancer = np.mean(d_cancer)  # Calculate the mean difference for the Breast Cancer dataset
    std_dev_cancer = np.std(d_cancer, ddof=1)  # Calculate the sample standard deviation for the Breast Cancer dataset

    mean_d_hr = np.mean(diff_hr)  # Calculate the mean difference for the Hayes-Roth dataset
    std_dev_hayes_roth = np.std(diff_hr, ddof=1)  # Calculate the sample standard deviation for the Hayes-Roth dataset

    # Number of folds (degrees of freedom)
    n_folds = len(knncar)  # Get the number of folds (assumes all datasets have the same number of folds)

    # Calculate t-statistic
    t_stat_car = (mean_d_car / (std_dev_car / np.sqrt(n_folds)))  # Calculate the t-statistic for the Car Evaluation dataset
    t_stat_cancer = (mean_d_cancer / (std_dev_cancer / np.sqrt(n_folds)))  # Calculate the t-statistic for the Breast Cancer dataset
    t_stat_hayes_roth = (mean_d_hr / (std_dev_hayes_roth / np.sqrt(n_folds)))  # Calculate the t-statistic for the Hayes-Roth dataset

    # Set the significance level (alpha)
    al = .99  # Set the significance level to 0.99 (99% confidence)

    # Calculate critical t-value
    t_c = stats.t.ppf(1 - al / 2, n_folds - 1)  # Calculate the critical t-value for the specified significance level and degrees of freedom

    # Determine if the differences are statistically significant
    def check_sc(t_stat, t_c):  # Define a function to check if the differences are statistically significant
        if abs(t_stat) > t_c:  # Compare the absolute t-statistic with the critical t-value
            return "It's Statistically significant (Null Hypothesis Rejected)"  # Return this message if the null hypothesis is rejected
        else:
            return "It's not statistically significant (Fail to Reject Null Hypothesis)"  # Return this message if the null hypothesis is not rejected

    # Perform t-test for each dataset
    print("\nResults from Paired T-test:")
    print("Evaluation - Car :", check_sc(t_stat_car, t_c))  # Check and print the result of the t-test for the Car Evaluation dataset
    print("Evaluation - Cancer:", check_sc(t_stat_cancer, t_c))  # Check and print the result of the t-test for the Breast Cancer dataset
    print("Evaluation - Hayes Roth:", check_sc(t_stat_hayes_roth, t_c))  # Check and print the result of the t-test for the Hayes-Roth dataset

