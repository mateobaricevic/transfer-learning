import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon
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
for n in range(12):
    # Load truths and predictions
    with open(f'{path}/model_{n}/truths.npy', 'rb') as f:
        truths = np.load(f)
    with open(f'{path}/model_{n}/predictions.npy', 'rb') as f:
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
    print('-' * 8 + f' Model {n} ' + '-' * 8)
    print(f'F1: {round(f1, 4)}')
    print(f'MCC: {round(mcc, 4)}')
    print(f'ROC AUC: {round(roc_auc, 4)}')  # type: ignore

# Calculate average scores
avg_f1s = np.average(f1s)
avg_mccs = np.average(mccs)
avg_roc_aucs = np.average(roc_aucs)

# Print average scores
print('\n' + '-' * 8 + f' Average Scores ' + '-' * 8)
print(f'F1: {round(avg_f1s, 4)}')
print(f'MCC: {round(avg_mccs, 4)}')
print(f'ROC AUC: {round(avg_roc_aucs, 4)}')

# Calculate standard deviations
std_f1s = np.std(f1s)
std_mccs = np.std(mccs)
std_roc_aucs = np.std(roc_aucs)

# Print standard deviations
print('\n' + '-' * 8 + f' Standard Deviations ' + '-' * 8)
print(f'F1: {round(std_f1s, 4)}')
print(f'MCC: {round(std_mccs, 4)}')
print(f'ROC AUC: {round(std_roc_aucs, 4)}')

# Save evaluation
print('\nSaving evaluations...')
with open(f'{path}/evaluation.json', 'w') as f:
    json.dump({
            'f1s': f1s,
            'mccs': mccs,
            'roc_aucs': roc_aucs,
            'avg_f1s': avg_f1s,
            'avg_mccs': avg_mccs,
            'avg_roc_aucs': avg_roc_aucs,
            'std_f1s': std_f1s,
            'std_mccs': std_mccs,
            'std_roc_aucs': std_roc_aucs,
        }, f, ensure_ascii=False, indent=4)
print('Done.')

if args.source_dataset:
    evaluation_path = f'results/{args.target_dataset}/evaluation.json'
    if not os.path.exists(evaluation_path):
        print(f'\nEvaluation for {args.target_dataset} is missing!')
        print(f'Please run "python evaluate.py {args.target_dataset}" first.')
        exit()
    
    with open(evaluation_path, 'r') as evaluation_file:
        evaluation = json.load(evaluation_file)
        
        # Calculate statistical test for each metric
        wilcoxon_f1 = wilcoxon(evaluation['f1s'], f1s).pvalue
        wilcoxon_mcc = wilcoxon(evaluation['mccs'], mccs).pvalue
        wilcoxon_roc_auc = wilcoxon(evaluation['roc_aucs'], roc_aucs).pvalue
        
        # Print statistical test
        print('\n' + '-' * 8 + f' Wilcoxon Signed-Rank Test ' + '-' * 8)
        print(f'F1: {round(wilcoxon_f1, 4)}')
        print(f'MCC: {round(wilcoxon_mcc, 4)}')
        print(f'ROC AUC: {round(wilcoxon_roc_auc, 4)}')

        # Save statistical test
        print('\nSaving statistical test...')
        with open(f'{path}/statistical_test.json', 'w') as f:
            json.dump({
                    'f1': wilcoxon_f1,
                    'f1s': {
                        'models': evaluation['f1s'],
                        'tl_models': f1s,
                    },
                    'mcc': wilcoxon_mcc,
                    'mccs': {
                        'models': evaluation['mccs'],
                        'tl_models': mccs,
                    },
                    'roc_auc': wilcoxon_roc_auc,
                    'roc_aucs': {
                        'models': evaluation['roc_aucs'],
                        'tl_models': roc_aucs,
                    },
                }, f, ensure_ascii=False, indent=4)
        print('Done.')
