import os

import librosa
import numpy as np

classes = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 
    'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
    'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 
    'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'
]

X = []
y = []

# Load all audio files and determine the length of the longest recording
max_len = 0
for index, c in enumerate(classes):
    print(c)
    files = os.listdir(f'datasets/speech/{c}/')
    for f in files:
        data, samplerate = librosa.load(f'datasets/speech/{c}/{f}', sr=8000)
        X.append(data)
        y.append(index)
        max_len = max(len(data), max_len)
print(f'Max length: {max_len}')

# Load background noise and pad all recordings to the length of the longest recording
white_noise, samplerate = librosa.load('datasets/speech/_background_noise_/white_noise.wav', sr=8000)
white_noise_i = 0
for i, x in enumerate(X):
    while len(x) != max_len:
        delta_len = max_len - len(x)
        indices = range(white_noise_i, white_noise_i + delta_len)
        x = np.append(x, white_noise.take(indices, mode='wrap'))
        white_noise_i = (white_noise_i + delta_len) % len(white_noise)
    X[i] = x

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

np.save('data/X_speech.npy', X)
np.save('data/y_speech.npy', y)
