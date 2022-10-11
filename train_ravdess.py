import csv
import pickle

import numpy as np
from keras.callbacks import EarlyStopping
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, train_test_split

import Models

classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Open numpy data
X = np.load('data/X_ravdess.npy')
y = np.load('data/y_ravdess.npy')

# Determine input size
waveform_length = len(X[0])

# Determine output size
num_classes = len(classes)

# Standardise data
X -= np.mean(X)
X /= np.std(X)

# Shuffle and separate data into train, validation and test data
stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)

for split_index, (train_indexes, test_indexes) in enumerate(stratified_k_fold.split(X, y)):
    # Seperate data into train and test data
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

    model.compile(
        optimizer=Adam(learning_rate=0.01),
        loss=categorical_crossentropy,
        metrics=['accuracy']
    )

    # Print model summary
    if split_index == 0:
        print(model.summary())

    # Fit model
    print(f'Fitting model {split_index}...')
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_validation, y_validation),
        epochs=250,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10)
        ]
    )
    
    # Save model and training history
    with open(f'models/ravdess_{split_index}.history', 'wb') as f:
        pickle.dump(history.history, f)
    model.save(f'models/ravdess_{split_index}.h5')
    
    # Predict using test data
    predictions = model.predict(x=X_test)
    
    # Save predictions
    with open(f'models/predictions/ravdess_{split_index}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for i, prediction in enumerate(predictions):
            writer.writerow([y_test[i], prediction])
    
