import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense


def datos(freq):
    computed_data = pd.read_csv(
        'DATA/labeled_features/features_' + freq + '.csv', parse_dates=['date'])
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

    X = np.array(computed_data[features])
    Y = np.array(computed_data['gt'].astype(int).values.ravel())
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.4, random_state=42)
    return X, Y, X_train, X_val, y_train, y_val


def model1():
    # Define the CNN model
    model = Sequential()
    # Add the dense layer
    model.add(Dense(128, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[tf.keras.metrics.Recall()])
    # Define the number of folds and the data to use for cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Initialize a list to store the scores
    scores = []
    # Train and evaluate the model with K-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print("Fold:", fold+1)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]

        model.fit(X_train_fold, y_train_fold,
                  epochs=10, batch_size=32, verbose=0)
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print("Validation recall:", score[1])
        scores.append(score[1])
    print('Model 1')
    # Compute the average score
    avg_score = np.mean(scores)
    print(avg_score)


def model2():
    # Define the CNN model
    model = Sequential()
    # Add the dense layer
    model.add(Dense(64, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[tf.keras.metrics.Recall()])
    # Define the number of folds and the data to use for cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Initialize a list to store the scores
    scores = []
    # Train and evaluate the model with K-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print("Fold:", fold+1)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]

        model.fit(X_train_fold, y_train_fold,
                  epochs=10, batch_size=32, verbose=0)
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print("Validation recall:", score[1])
        scores.append(score[1])
    print('Model 2')
    # Compute the average score
    avg_score = np.mean(scores)
    print(avg_score)


def model3():
    # Define the CNN model
    model = Sequential()
    # Add the dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))
    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[tf.keras.metrics.Recall()])
    # Define the number of folds and the data to use for cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Initialize a list to store the scores
    scores = []
    # Train and evaluate the model with K-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print("Fold:", fold+1)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]

        model.fit(X_train_fold, y_train_fold,
                  epochs=10, batch_size=32, verbose=0)
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print("Validation recall:", score[1])
        scores.append(score[1])
    print('Model 3')
    # Compute the average score
    avg_score = np.mean(scores)
    print(avg_score)


def model4():
    # Define the CNN model
    model = Sequential()

    # Add the dense layer
    model.add(Dense(128, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[tf.keras.metrics.Recall()])
    # Define the number of folds and the data to use for cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Initialize a list to store the scores
    scores = []
    # Train and evaluate the model with K-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print("Fold:", fold+1)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]

        model.fit(X_train_fold, y_train_fold,
                  epochs=10, batch_size=32, verbose=0)
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print("Validation recall:", score[1])
        scores.append(score[1])
    print('Model 4')
    # Compute the average score
    avg_score = np.mean(scores)
    print(avg_score)


def model5():
    # Define the CNN model
    model = Sequential()

    # Add the dense layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Add the output layer
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adam,
                  metrics=[tf.keras.metrics.Recall()])
    # Define the number of folds and the data to use for cross-validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Initialize a list to store the scores
    scores = []
    # Train and evaluate the model with K-fold cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y)):
        print("Fold:", fold+1)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = Y[train_index], Y[val_index]

        model.fit(X_train_fold, y_train_fold,
                  epochs=15, batch_size=32, verbose=0)
        score = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print("Validation recall:", score[1])
        scores.append(score[1])
    print('Model 5')
    # Compute the average score
    avg_score = np.mean(scores)
    print(avg_score)
    model.save('./DNN5S.h5')


if __name__ == '__main__':
   X, Y, X_train, X_val, y_train, y_val = datos('5s')
   #model1()
   #model2()
   #model3()
   #model4()
   model5()
