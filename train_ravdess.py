import os
import os.path
import pickle

import numpy as np
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

import Models

os.environ['TF_KERAS'] = '1'

train_percentage = 75
validation_percentage = 15
test_percentage = 15
classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Open specified dataset
X = np.load('data/ravdess_X.npy')
y = np.load('data/ravdess_y.npy')

# Shuffle data
permutation = np.random.permutation(len(X))
X = X[permutation]
y = y[permutation]

# Determine waveform length
waveform_length = len(X[0])

# Determine output size
num_classes = len(classes)

# Convert index to array of binary values representing a class
y = np.eye(num_classes)[y]

# Separate data into train, validation, test
train_X, validation_X, test_X = np.split(X, [int(len(X) * 0.70), int(len(X) * 0.85)])
train_y, validation_y, test_y = np.split(y, [int(len(y) * 0.70), int(len(y) * 0.85)])

# Get and compile model
model = Models.magnet(waveform_length, num_classes)

model.compile(
    optimizer=Adam(learning_rate=0.01), 
    loss=categorical_crossentropy, 
    metrics=['accuracy']
)

# Print model summary
print(model.summary())

# Fit model
history = model.fit(
    x=train_X,
    y=train_y,
    epochs=25
)
  
# Save model and training history
with open('models/ravdess.history', 'wb') as f:
    pickle.dump(history.history, f)
model.save('models/ravdess.h5')
