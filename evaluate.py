import argparse

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="which model to evaluate", required=True)
args = parser.parse_args()

f1s = []
mccs = []
for i in range(4):
    with open(f'models/predictions/{args.model}_{i}_truths.npy', 'rb') as f:
        truths = np.load(f)
    with open(f'models/predictions/{args.model}_{i}_predictions.npy', 'rb') as f:
        predictions = np.load(f)
    
    truths_indexes = [np.argmax(truth) for truth in truths]
    predictions_indexes = [np.argmax(prediction) for prediction in predictions]

    f1 = f1_score(truths_indexes, predictions_indexes, average='weighted')
    f1s.append(f1)
    mcc = matthews_corrcoef(truths_indexes, predictions_indexes)
    mccs.append(mcc)
    
    print('-' * 8 + f' Model {i} ' + '-' * 8)
    print(f'F1 score: {f1}')
    print(f'Matthews correlation coefficient: {mcc}')

print('-' * 8 + f' Average Scores ' + '-' * 8)
print(f'F1 score: {np.average(f1s)}')
print(f'Matthews correlation coefficient: {np.average(mccs)}')
