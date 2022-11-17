import argparse

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('target_dataset', help='Which dataset was used for training?')
parser.add_argument('source_dataset', help='Which dataset was knowledge transfered from?', nargs='?', default=None)
parser.add_argument('n_layers', help='How many layers of weights were transfered?', nargs='?', type=int, default=None)
args = parser.parse_args()

# Indexes of layers that can be used to transfer knowledge from
layers = [1, 4, 7, 10, 13]

if args.source_dataset:
    if args.n_layers is None:
        parser.error('when providing source_dataset argument the following arguments are also required: n_layers')
    if args.n_layers < 1 or args.n_layers > len(layers):
        parser.error(f'argument n_layers: invalid value (needs to be from range [1, {len(layers)}])')

# Get path
path = f'results/{args.target_dataset}'
if args.source_dataset:
    path += f'/{args.source_dataset}/{args.n_layers}_layers'

# Evaluate models
f1s = []
mccs = []
roc_aucs = []
for n_fold in range(4):
    # Load truths and predictions
    with open(f'{path}/model_{n_fold}/truths.npy', 'rb') as f:
        truths = np.load(f)
    with open(f'{path}/model_{n_fold}/predictions.npy', 'rb') as f:
        predictions = np.load(f)

    truths_indexes = [np.argmax(truth) for truth in truths]
    predictions_indexes = [np.argmax(prediction) for prediction in predictions]

    # Calculate F1 score
    f1 = f1_score(truths_indexes, predictions_indexes, average='weighted')
    f1s.append(f1)
    
    # Calculate Matthews correlation coefficient
    mcc = matthews_corrcoef(truths_indexes, predictions_indexes)
    mccs.append(mcc)
    
    # Calculate Area Under the Receiver Operating Characteristic Curve score
    roc_auc = roc_auc_score(truths, predictions, average='weighted')
    roc_aucs.append(roc_auc)

    # Print scores
    print('-' * 8 + f' Model {n_fold} ' + '-' * 8)
    print(f'F1 score: {f1}')
    print(f'Matthews correlation coefficient: {mcc}')
    print(f'ROC AUC score: {roc_auc}')

# Print average scores
print()
print('-' * 8 + f' Average Scores ' + '-' * 8)
print(f'F1 score: {np.average(f1s)}')
print(f'Matthews correlation coefficient: {np.average(mccs)}')
print(f'ROC AUC score: {np.average(roc_aucs)}')

# Print standard deviations
print()
print('-' * 8 + f' Standard Deviations ' + '-' * 8)
print(f'F1 score: {np.std(f1s)}')
print(f'Matthews correlation coefficient: {np.std(mccs)}')
print(f'ROC AUC score: {np.std(roc_aucs)}')
