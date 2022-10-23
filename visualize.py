import argparse
import pickle

from matplotlib import pyplot as plt

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
path = f'models/{args.target_dataset}'
if args.source_dataset:
    path += f'/{args.source_dataset}_{args.n_layers}'

if_source_dataset = ''
if args.source_dataset:
    if_source_dataset = f' using {args.n_layers} layer(s) of knowledge from {args.source_dataset} models'

# Vizualize models
print(f'Visualizing {args.target_dataset} models{if_source_dataset}...')
for i in range(4):
    # Load training history
    with open(f'{path}/model_{i}.history', 'rb') as f:
        history = pickle.load(f)

    # Get training accuracies
    accuracy = history['accuracy'][:-8]
    val_accuracy = history['val_accuracy'][:-8]

    # Get training losses
    loss = history['loss'][:-8]
    val_loss = history['val_loss'][:-8]

    # Save number of epochs before finetuning
    n_epochs = len(loss)

    if args.source_dataset:
        # Load finetuning history
        with open(f'{path}/model_{i}_finetune.history', 'rb') as f:
            ft_history = pickle.load(f)

            # Append finetuning accuracies
            accuracy += ft_history['accuracy'][:-8]
            val_accuracy += ft_history['val_accuracy'][:-8]

            # Append finetuning losses
            loss += ft_history['loss'][:-8]
            val_loss += ft_history['val_loss'][:-8]

    # Get epochs
    epochs = range(1, len(loss) + 1)

    # Create figure
    figure = plt.figure(figsize=(12, 8))

    # Plot training and validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(epochs, accuracy, color='cornflowerblue', label='Training accuracy')
    plt.plot(epochs, val_accuracy, color='orange', label='Validation accuracy')
    if args.source_dataset:
        plt.axvline(n_epochs, color='limegreen', label='Finetuning started')
    plt.xticks(epochs)
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Plot training and validation loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, loss, color='cornflowerblue', label='Training loss')
    plt.plot(epochs, val_loss, color='orange', label='Validation loss')
    if args.source_dataset:
        plt.axvline(n_epochs, color='limegreen', label='Finetuning started')
    plt.xticks(epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Show figure
    # plt.show()

    # Save figure
    print(f'Saving figure for model_{i}...')
    figure.savefig(f'{path}/model_{i}.png')

print('Done.')
