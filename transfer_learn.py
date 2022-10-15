import argparse
import pickle

import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import Models

parser = argparse.ArgumentParser()
parser.add_argument('source_dataset', help='Which dataset to transfer knowledge from?')
parser.add_argument('target_dataset', help='Which dataset to transfer knowledge to?')
args = parser.parse_args()

# Load data from target dataset
print(f'Loading data from {args.target_dataset}...')
classes = np.load(f'data/{args.target_dataset}/classes.npy')
X = np.load(f'data/{args.target_dataset}/X.npy')
y = np.load(f'data/{args.target_dataset}/y.npy')

# Determine input size
waveform_length = len(X[0])

# Determine output size
num_classes = len(classes)

for i in range(4):
    # Load pretrained model of source dataset
    print(f'Loading pretrained model {i} of {args.source_dataset}...')
    pretrained_model = keras.models.load_model(f'models/{args.source_dataset}/model_{i}.h5')

    if pretrained_model is None:
        raise Exception('Failed to load pretrained model!')

    # Print pretrained model summary
    if i == 0:
        print('-' * 8 + ' Source model ' + '-' * 8)
        print(pretrained_model.summary())

    # Get model
    model = Models.magnet(waveform_length, num_classes)

    # Transfer knowledge
    for j in range(2):
        model.layers[j].set_weights(pretrained_model.layers[j].get_weights())
        model.layers[j].trainable = False

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    # Print model summary
    if i == 0:
        print('-' * 8 + ' Target model ' + '-' * 8)
        print(model.summary())

    # Split target data into train, validation and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=64)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=64)

    # Convert index to array of binary values representing a class
    y_train = np.eye(num_classes)[y_train]
    y_validation = np.eye(num_classes)[y_validation]
    y_test = np.eye(num_classes)[y_test]

    # Fit model
    print(f'Fitting model_{i}...')
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
    model.save(f'models/{args.target_dataset}/{args.source_dataset}/model_{i}.h5')
    with open(f'models/{args.target_dataset}/{args.source_dataset}/model_{i}.history', 'wb') as f:
        pickle.dump(history.history, f)
    with open(f'models/{args.target_dataset}/{args.source_dataset}/predictions/model_{i}_truths.npy', 'wb') as f:
        np.save(f, y_test)
    with open(f'models/{args.target_dataset}/{args.source_dataset}/predictions/model_{i}_predictions.npy', 'wb') as f:
        np.save(f, predictions)

print('Done.')
