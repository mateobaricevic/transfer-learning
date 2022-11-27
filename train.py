import argparse
import os
import pickle
from datetime import datetime

import numpy as np
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

import Models

parser = argparse.ArgumentParser()
parser.add_argument('target_dataset', help='Which dataset to train?')
parser.add_argument('source_dataset', help='Which dataset to transfer knowledge from?', nargs='?', default=None)
parser.add_argument('n_layers', help='How many layers of weights to transfer?', nargs='?', default=None, type=int)
parser.add_argument('-f', '--finetune', help='Finetune the model?', action='store_true')
args = parser.parse_args()

# Indexes of layers that can be used to transfer knowledge from
layers = [1, 4, 7, 10, 13]

if args.source_dataset:
    if args.n_layers is None:
        parser.error('when providing source_dataset argument the following arguments are also required: n_layers')
    if args.n_layers < 1 or args.n_layers > len(layers):
        parser.error(f'argument n_layers: invalid value (needs to be from range [1, {len(layers)}])')

# Load data
print('Loading data...')
classes = np.load(f'datasets/preprocessed/{args.target_dataset}/classes.npy')
X = np.load(f'datasets/preprocessed/{args.target_dataset}/X.npy')
y = np.load(f'datasets/preprocessed/{args.target_dataset}/y.npy')

# Determine input size
waveform_length = len(X[0])

# Determine output size
num_classes = len(classes)

# Standardize data
X -= np.mean(X)
X /= np.std(X)

# Get path
path = f'results/{args.target_dataset}'
if args.source_dataset:
    path += f'/{args.source_dataset}/{args.n_layers}_layers'
    
# Split data into train and test stratified data and use k-fold cross-validation
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=64)

for n, (train_indexes, test_indexes) in enumerate(rskf.split(X, y)):
    # Get train and test data
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    # Seperate train data into train and validation data
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=64)

    # Convert index to array of binary values representing a class
    y_train = np.eye(num_classes)[y_train]
    y_validation = np.eye(num_classes)[y_validation]
    y_test = np.eye(num_classes)[y_test]

    # Get model
    model = Models.magnet(waveform_length, num_classes)

    if args.source_dataset:
        # Load pretrained model
        print(f'Loading pretrained model_{n} of {args.source_dataset}...')
        pretrained_model = load_model(f'results/{args.source_dataset}/model_{n}/model.h5')

        if pretrained_model is None:
            raise Exception(f'Pretrained model_{n} of {args.source_dataset} not found!')

        # Print pretrained model summary
        if n == 0:
            print('-' * 8 + ' Pretrained model ' + '-' * 8)
            print(pretrained_model.summary())

        # Transfer knowledge
        for i in layers[:args.n_layers]:
            pretrained_weights = pretrained_model.layers[i].get_weights()
            model.layers[i].set_weights(pretrained_weights)
            model.layers[i].trainable = False

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

    # Print model summary
    if n == 0:
        print('-' * 8 + ' Model ' + '-' * 8)
        print(model.summary())

    # Fit model
    print(f'Fitting model_{n} to {args.target_dataset}...')
    start_time = datetime.now()
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_validation, y_validation),
        epochs=128,
        callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
    )
    print(f'Training took {datetime.now() - start_time}.')
    
    # Make directory if it doesn't exist
    os.makedirs(f'{path}/model_{n}', exist_ok=True)

    # Save training history
    print('Saving training history...')
    with open(f'{path}/model_{n}/train.history', 'wb') as f:
        pickle.dump(history.history, f)

    if args.source_dataset and args.finetune:
        # Finetune model
        print(f'Finetuning model_{n}...')
        model.trainable = True
        model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])
        start_time = datetime.now()
        history = model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_validation, y_validation),
            epochs=128,
            callbacks=[EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)],
        )
        print(f'Finetuning took {datetime.now() - start_time}.')

        # Save finetuning history
        print('Saving finetuning history...')
        with open(f'{path}/model_{n}/finetune.history', 'wb') as f:
            pickle.dump(history.history, f)

    # Predict using test data
    print('Predicting using test data...')
    predictions = model.predict(x=X_test)

    # Save model, truths and predictions
    print('Saving model, truths and predictions...')
    model.save(f'{path}/model_{n}/model.h5')
    with open(f'{path}/model_{n}/truths.npy', 'wb') as f:
        np.save(f, y_test)
    with open(f'{path}/model_{n}/predictions.npy', 'wb') as f:
        np.save(f, predictions)

print('Done.')
