import argparse
import os
parser = argparse.ArgumentParser()

# Data
parser.add_argument('--data', type=str)
parser.add_argument('--nviews', type=int)
parser.add_argument('--sampling', choices=['ib', 'cb', 'cr'])

# Model
parser.add_argument('--model', type=str)
parser.add_argument('--load', action='store_true')
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
parser.add_argument('--pce', action='store_true')
parser.add_argument('--temp', type=float, default=0.1)

# Save
parser.add_argument('--out-dir', type=str)


import data
import plots
import models
import train

def run(args):
  # Folder to save all work
  if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

  # Data
  train_loader, test_loader = data.load_data(args)

  # Display sample of images from train loader
  train_samples, _ = next(iter(train_loader))
  plots.plot_samples(train_samples[0][:8])

  # Model
  model = models.make_model(args)

  # Optionally load weights
  models.optional_load_wts(args, model, f'{args.out_dir}/model.pt')

  # Classifier reset train?
  if args.crt:
    model.reset_classifier()
    print('Reset classifier')

  # Train
  metrics = train.train(args, model, train_loader, test_loader)

  # Plot training metrics
  if args.epochs > 0:
    plots.plot_metrics(metrics, args.out_dir)

  # Plot metrics on features
  if args.tsne:
    plots.plot_tsne_similarity_types(model, train_loader, f'{args.out_dir}/tsne_sim_types.jpg')
    plots.plot_tsne(model, test_loader, f'{args.out_dir}/tsne.jpg')
  if args.contrast:
    plots.plot_similarity_hist(model, train_loader, f'{args.out_dir}/sims.jpg')

  print(f'models and plots saved to {args.out_dir}')

if __name__ == '__main__':
  args = parser.parse_args()
  print(args)
  run(args)