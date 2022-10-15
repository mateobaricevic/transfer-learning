import argparse
import pickle

import numpy as np
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split

import Models

parser = argparse.ArgumentParser()
parser.add_argument('dataset', help='Which dataset to train?')
args = parser.parse_args()

# Load data
print('Loading data...')
classes = np.load(f'data/{args.dataset}/classes.npy')
X = np.load(f'data/{args.dataset}/X.npy')
y = np.load(f'data/{args.dataset}/y.npy')

# Determine input size
waveform_length = len(X[0])

# Determine output size
num_classes = len(classes)

# Standardize data
X -= np.mean(X)
X /= np.std(X)

# Split data into train and test stratified data and use KFold cross-validation
stratified_k_fold = StratifiedKFold(n_splits=4, shuffle=True, random_state=64)

for split_index, (train_indexes, test_indexes) in enumerate(stratified_k_fold.split(X, y)):
    # Get train and test data
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    # Seperate train data into train and validation data
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=64)

    # Convert index to array of binary values representing a class
    y_train = np.eye(num_classes)[y_train]
    y_validation = np.eye(num_classes)[y_validation]
    y_test = np.eye(num_classes)[y_test]

    # Get and compile model
    model = Models.magnet(waveform_length, num_classes)

    model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    # Print model summary
    if split_index == 0:
        print(model.summary())

    # Fit model
    print(f'Fitting model_{split_index}...')
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_validation, y_validation),
        epochs=128,
        callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
    )

    # Predict using test data
    print('Predicting using test data...')
    predictions = model.predict(x=X_test)

    # Save model, training history, truths and predictions
    print('Saving model, history, truths and predictions...')
    model.save(f'models/{args.dataset}/model_{split_index}.h5')
    with open(f'models/{args.dataset}/model_{split_index}.history', 'wb') as f:
        pickle.dump(history.history, f)
    with open(f'models/{args.dataset}/predictions/model_{split_index}_truths.npy', 'wb') as f:
        np.save(f, y_test)
    with open(f'models/{args.dataset}/predictions/model_{split_index}_predictions.npy', 'wb') as f:
        np.save(f, predictions)

print('Done.')
