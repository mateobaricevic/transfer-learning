import os

import librosa
import numpy as np

classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
X = []
y = []

# Load all audio files and determine the length of the longest recording
max_len = 0
for index in range(24):
    actor = f'Actor_{str(index + 1).zfill(2)}'
    print(f'Processing data from {actor}...')
    files = os.listdir(f'datasets/ravdess/{actor}/')
    for f in files:
        data, samplerate = librosa.load(f'datasets/ravdess/{actor}/{f}', sr=8000)
        X.append(data)
        y.append(int(f[6:8]) - 1)
        max_len = max(len(data), max_len)

print(f'Max waveform length: {max_len}')

# Load background noise and pad all recordings to the length of the longest recording
print('Padding data with white noise...')
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
classes = np.array(classes)
X = np.array(X)
y = np.array(y)

# Save data
print('Saving data...')
np.save('data/ravdess/classes.npy', classes)
np.save('data/ravdess/X.npy', X)
np.save('data/ravdess/y.npy', y)

print('Done.')
