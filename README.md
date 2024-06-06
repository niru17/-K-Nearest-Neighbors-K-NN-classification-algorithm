# K-Nearest-Neighbors-K-NN-classification-algorithm
This code implements a k-Nearest Neighbors (k-NN) classification algorithm from scratch, compares its performance with scikit-learn's k-NN implementation using 10-fold cross-validation, and performs a paired t-test to check for statistical significance in performance differences.

Breakdown of the Code:

Importing Libraries:
- numpy: For numerical operations.
- pandas: For data manipulation (not used in the code though).
- csv: For reading CSV files.
- random: For generating random numbers.
- math: For mathematical operations.
- scikit-learn: For k-NN classifier and cross-validation.

Loading and Converting Data:
- loadd(f): Loads a CSV file into a list of rows.
- dataset(f): Wrapper function to call loadd(f).
- convert_n(d): Converts categorical data in the dataset to numerical data by replacing categories with their index.
- convert2numeric(d): Wrapper function to call convert_n(d).

Cross-Validation:

- valid(d, f): Splits the dataset d into f folds.
- val_cross(d, f=10): Wrapper function to call valid(d, f).
  
Distance Calculation:

- euclid(p_1, p_2): Calculates Euclidean distance between two data points.
- dist_euclid(p_1, p_2): Wrapper function to call euclid(p_1, p_2).

Finding Neighbors:

- neigh(train, testins, n): Finds the n nearest neighbors of a test instance testins from the training set train using Euclidean distance.
- neighbors_g(train, testins, k): Wrapper function to call neigh(train, testins, n).

Making Predictions:

- prediction(train, testins, c): Predicts the class of a test instance testins based on c nearest neighbors.
- classpredict(train, testins, c): Wrapper function to call prediction(train, testins, c).

Evaluating k-NN:

- getknn(d, k, f=10): Evaluates the k-NN algorithm using f-fold cross-validation on dataset d with k neighbors.
- knn(d, k, f=10): Wrapper function to call getknn(d, k, f).

Accuracy Calculation:

- acc(a, p): Calculates the accuracy percentage.
- calc_acc(a, p): Wrapper function to call acc(a, p).

Preprocessing Datasets:

- preprocess_car(): Loads and preprocesses the Car Evaluation dataset.
- preprocess_cancer(): Loads and preprocesses the Breast Cancer dataset.
- preprocess_hr(): Loads and preprocesses the Hayes-Roth dataset.

Main Execution:

- Loading and preprocessing datasets: Preprocesses the Car Evaluation, Breast Cancer, and Hayes-Roth datasets.
- Evaluating custom k-NN implementation: Uses 10-fold cross-validation to evaluate k-NN on the three datasets.
- Evaluating scikit-learn k-NN implementation: Uses cross_val_score for 10-fold cross-validation.
- Performing paired t-test: Compares the accuracy of the custom implementation with scikit-learn's implementation to determine statistical significance.
  
Example Run

The main script sets up and runs the above steps as follows:

- Load and preprocess datasets.
- Set k = 3 for k-NN.
- Evaluate custom k-NN implementation and print accuracy and mean accuracy for each dataset.
- Evaluate scikit-learn k-NN implementation and print accuracy and mean accuracy for each dataset.
- Perform paired t-test on accuracy scores from custom implementation and scikit-learn, and print whether the differences are statistically significant.
