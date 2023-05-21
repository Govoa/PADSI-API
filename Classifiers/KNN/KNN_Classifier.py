import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
import numpy as np
from joblib import dump, load
from sklearn.neighbors import KNeighborsClassifier


def KNN_Classifier(time_freq):

    computed_data = pd.read_csv('DATA/labeled_features/features_' + time_freq + '.csv', parse_dates=['date'])
    features = ['std_rush_order',
                'avg_rush_order',
                'std_trades',
                'std_volume',
                'avg_volume',
                'std_price',
                'avg_price',
                'avg_price_max',
                'hour_sin',
                'hour_cos',
                'minute_sin',
                'minute_cos']

    X = computed_data[features]
    Y = computed_data['gt'].astype(int).values.ravel()


    # Define the number of splits for K-fold cross-validation
    n_splits = 7

    # Define the SVM classifier
    clf =  KNeighborsClassifier(n_neighbors=5)


    # Define the K-fold cross-validator
    kf = KFold(n_splits=n_splits)

    # Create empty arrays to store the test accuracies, recalls, and F1 scores
    test_accuracies = []
    recalls = []
    f1_scores = []

    # Perform K-fold cross-validation
    for train_index, test_index in kf.split(X):
        # Split the data into training and test sets for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        # Train the model on the current training set
        clf.fit(X_train, y_train)
        
        # Evaluate the model on the current test set
        y_pred = clf.predict(X_test)
        
        # Compute and store the test accuracy, recall, and F1 score for the current fold
        test_accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        
        test_accuracies.append(test_accuracy)
        recalls.append(recall)
        f1_scores.append(f1)
        # save the model to disk
    dump(clf, 'KNN'+ time_freq + '.joblib')
    # Compute the mean and standard deviation of the test accuracies, recalls, and F1 scores over all folds
    mean_test_accuracy = np.mean(test_accuracies)
    std_test_accuracy = np.std(test_accuracies)
    mean_recall = np.mean(recalls)
    std_recall = np.std(recalls)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    # Print the mean and standard deviation of the test accuracies, recalls, and F1 scores over all folds
    print('Test accuracy (mean/std):', mean_test_accuracy, '/', std_test_accuracy)
    print('Recall (mean/std):', mean_recall, '/', std_recall)
    print('F1 score (mean/std):', mean_f1, '/', std_f1)

if __name__ == '__main__':
    start = datetime.datetime.now()
    KNN_Classifier(time_freq='25S')
    KNN_Classifier(time_freq='15S')
    KNN_Classifier(time_freq='5S')
    print(datetime.datetime.now() - start)