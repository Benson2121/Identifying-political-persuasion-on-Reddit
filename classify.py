#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel, stats
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

# set the random state for reproducibility 
import numpy as np

np.random.seed(401)

models = [SGDClassifier(), GaussianNB(), RandomForestClassifier(n_estimators=10, max_depth=5),
          MLPClassifier(alpha=0.05), AdaBoostClassifier()]


def accuracy(C):
    """ Compute accuracy given Numpy array confusion matrix C. Returns a floating point value """
    return np.sum(np.diag(C)) / np.sum(C)


def recall(C):
    """ Compute recall given Numpy array confusion matrix C. Returns a list of floating point values """
    return np.diag(C) / np.sum(C, axis=0)


def precision(C):
    """ Compute precision given Numpy array confusion matrix C. Returns a list of floating point values """
    return np.diag(C) / np.sum(C, axis=1)


def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    best_accuracy = 0
    iBest = -1

    # For write the printed statement into the file.
    model_strings = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier"]

    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:

        for i in range(0, 5):

            # Get the model
            model = models[i]
            # Train the model
            model.fit(X_train, y_train)
            # Make prediction
            y_prediction = model.predict(X_test)

            # Get confusion_matrix and calculate accuracy, recall, and precision.
            confusion_matrx = confusion_matrix(y_test, y_prediction)
            model_accuracy = accuracy(confusion_matrx)
            model_recall = recall(confusion_matrx)
            model_precision = precision(confusion_matrx)

            # Replace the best model if the model has higher accuracy.
            if model_accuracy > best_accuracy:
                best_accuracy = model_accuracy
                iBest = i

            classifier_name = model_strings[i]
            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {model_accuracy:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in model_recall]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in model_precision]}\n')
            outf.write(f'\tConfusion Matrix: \n{confusion_matrx}\n\n')

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:

        for data_amount in [1000, 5000, 10000, 15000, 20000]:

            # Sampling data
            sample_index = np.random.choice(X_train.shape[0], data_amount, replace=False)

            sample_X_train = X_train[sample_index]
            sample_y_train = y_train[sample_index]

            # Get model
            model = models[iBest]
            model.fit(sample_X_train, sample_y_train)

            # Make prediction and calculate accuracy
            y_prediction = model.predict(X_test)
            confusion_matrx = confusion_matrix(y_test, y_prediction)
            model_accuracy = accuracy(confusion_matrx)

            outf.write(f'{data_amount}: {model_accuracy:.4f}\n')

    return X_train[:1000], y_train[:1000]


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    """ This function performs experiment 3.3

    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    """

    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.

        # For each number of features k_feat, write the p-values for that number of features (32K dataset)
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit_transform(X_train, y_train)

            pp = selector.pvalues_
            p_values = np.sort(pp, axis=None)[:k_feat]

            outf.write(f'{k_feat} p-values (full dataset): {[format(p_val) for p_val in p_values]}\n')

            if k_feat == 5:
                top_5_32k = np.argsort(pp)[:k_feat]

        # For each number of features k_feat, write the p-values for that number of features (1K dataset)
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k=k_feat)
            selector.fit_transform(X_1k, y_1k)

            pp = selector.pvalues_
            p_values = np.sort(pp, axis=None)[:k_feat]

            outf.write(f'{k_feat} p-values (1k dataset): {[format(p_val) for p_val in p_values]}\n')

            if k_feat == 5:
                top_5_1k = np.argsort(pp)[:k_feat]

        # 1K Dataset
        model = models[i]
        model.fit(X_1k[:, top_5_1k], y_1k)
        prediction_1k = model.predict(X_test[:, top_5_1k])

        accuracy_1k = accuracy(confusion_matrix(y_test, prediction_1k))
        outf.write(f'Accuracy for 1k dataset: {accuracy_1k:.4f}\n')

        # 32K Dataset
        model = models[i]
        model.fit(X_train[:, top_5_32k], y_train)
        prediction_32k = model.predict(X_test[:, top_5_32k])

        accuracy_32k = accuracy(confusion_matrix(y_test, prediction_32k))
        outf.write(f'Accuracy for full dataset: {accuracy_32k:.4f}\n')

        outf.write(f'Top-5 features indices for full dataset: {top_5_32k}\n')
        outf.write(f'Top-5 features indices for 1k dataset: {top_5_1k}\n')

        # Intersection
        intersection = list(set(top_5_1k) & set(top_5_32k))

        outf.write(f'Chosen feature intersection: {intersection}\n')


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    """ This function performs experiment 3.4

    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
    """

    n_splits = 5
    num_models = 5

    X_data = np.concatenate([X_train, X_test])
    y_data = np.concatenate([y_train, y_test])

    # Cross-Validation
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        k_fold_accuracies = np.zeros(shape=(num_models, n_splits))

        k_fold = KFold(n_splits=n_splits, random_state=401, shuffle=True)

        # Train the model based on the training dataset, and test based on the testing dataset.
        k = 0
        for (train_idx, test_idx) in k_fold.split(X_data):
            j = 0
            for model in models:
                X_fold_train, X_fold_test = X_data[train_idx], X_data[test_idx]
                y_fold_train, y_fold_test = y_data[train_idx], y_data[test_idx]

                model.fit(X_fold_train, y_fold_train)
                y_fold_prediction = model.predict(X_fold_test)
                confusion_matrx = confusion_matrix(y_fold_test, y_fold_prediction)
                k_fold_accuracies[j, k] = accuracy(confusion_matrx)
                j += 1
            k += 1

        # Computer accuracies
        for index in range(k):
            fold_lst = [round(acc, 4) for acc in k_fold_accuracies[:, index]]
            outf.write(f'Kfold Accuracies: {fold_lst} (fold {index + 1})\n')
            outf.write(f'\tMean of {index + 1}_fold accuracies is: {np.mean(fold_lst)}')

        # Determine whether the best model is significantly better than others.
        p_values = []
        for index in range(5):
            if index != i:
                p_value = stats.ttest_rel(k_fold_accuracies[index, :], k_fold_accuracies[i, :]).pvalue
                p_values.append(p_value)
        outf.write(f'p-values: {[format(pval) for pval in p_values]}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()

    # Load Data
    data = np.load(args.input)['arr_0']

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(data[:, :173], data[:, 173], test_size=0.2)

    # Find Best Classifiers
    iBest = class31(args.output_dir, X_train, X_test, y_train, y_test)

    # Amount of training data
    X_train_1k, y_train_1k = class32(args.output_dir, X_train, X_test, y_train, y_test, iBest)

    # Feature analysis
    class33(args.output_dir, X_train, X_test, y_train, y_test, iBest, X_train_1k, y_train_1k)

    # Cross-validation
    class34(args.output_dir, X_train, X_test, y_train, y_test, iBest)