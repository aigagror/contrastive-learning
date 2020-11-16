import argparse
import os

from data import load_data
from metrics import plot_samples, plot_metrics, plot_similarity_hist, plot_tsne
from models import make_model, optional_load_wts
from train import train

parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', type=str)
parser.add_argument('--sampling', choices=['ib', 'cb', 'cr'], default='ib')
parser.add_argument('--mi-lt-train', default="/gdrive/My Drive/datasets/miniimagenet/custom-lt/train.pkl")
parser.add_argument('--mi-lt-test', default="/gdrive/My Drive/datasets/miniimagenet/custom-lt/test.pkl")

# Model
parser.add_argument('--model', type=str)
parser.add_argument('--new', action='store_true')
parser.add_argument('--fix-feats', action='store_true')
parser.add_argument('--crt', action='store_true')

# Train
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--steplr', type=int, default=None)
parser.add_argument('--cosine', action='store_true')
parser.add_argument('--batchsize', type=int)

parser.add_argument('--tsne', action='store_true')

# Contrastive learning
parser.add_argument('--contrast', action='store_true')
parser.add_argument('--temp', type=float, default=0.1)

# Save
parser.add_argument('--out-dir', type=str)


def main(args):
    # Folder to save all work
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # Data
    train_loader, test_loader = load_data(args)

    # Display sample of images from train loader
    train_samples, _ = next(iter(train_loader))
    plot_samples(train_samples[0][:8])

    # Model
    model = make_model(args)

    # Optionally load weights
    model_path = f'{args.out_dir}/model.pt'
    optional_load_wts(args, model, model_path)

    # Classifier reset train?
    if args.crt:
        model.reset_classifier()
        print('Reset classifier')

    # Train
    metrics = train(args, model, train_loader, test_loader, model_path)

    # Plot training metrics
    if args.epochs > 0:
        plot_metrics(metrics, args.out_dir)

    # Plot metrics on features
    if args.contrast:
        plot_similarity_hist(model, train_loader, f'{args.out_dir}/sims.jpg')
    if args.tsne:
        plot_tsne(model, test_loader, f'{args.out_dir}/tsne.jpg')

    print(f'models and plots saved to {args.out_dir}')


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
