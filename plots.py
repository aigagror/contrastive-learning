import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid


## Metrics

class Metrics:
    def __init__(self):
        self.losses = {'train': [], 'test': []}
        self.label = {'train': [], 'test': []}
        self.pred = {'train': [], 'test': []}

    def epoch_append_data(self, subset, loss, label, pred):
        self.losses[subset].append(loss)
        self.label[subset].append(label)
        self.pred[subset].append(pred)

    def epoch_loss(self, subset, reduce_steps):
        losses = np.array(self.losses[subset])
        assert losses.ndim == 2
        n_epochs = len(losses)
        if reduce_steps:
            losses = np.mean(losses, axis=1)
        else:
            losses = losses.flatten()

        return losses, n_epochs

    def epoch_acc(self, subset):
        label = np.array(self.label[subset])
        pred = np.array(self.pred[subset])
        acc = np.mean(label == pred, axis=1)
        n_epochs = len(acc)
        return acc, n_epochs

    def last_epoch_acc(self, subset):
        """
        Returns instance-balanced accuracy and individual class accuracy
        """
        label = np.array(self.label[subset][-1])
        pred = np.array(self.pred[subset][-1])
        correct = label == pred
        class_acc = []
        for c in np.unique(label):
            class_acc.append(np.mean(correct[label == c]))
        return np.mean(correct), np.array(class_acc)

    def epoch_str(self):
        all_loss = []
        train_accs = self.last_epoch_acc('train')
        test_accs = self.last_epoch_acc('test')
        for subset in ['train', 'test']:
            all_loss.append(np.mean(self.losses[subset][-1]))

        epoch_str = '(train/test): ' \
                    f'({train_accs[0]:.3}/{train_accs[1].mean():.3})/({test_accs[0]:.3}/{test_accs[1].mean():.3}) inst/class acc, ' \
                    f'{all_loss[0]:.3}/{all_loss[1]:.3} loss'
        return epoch_str

    def todict(self):
        return self.__dict__.copy()


def plot_samples(train_samples):
    grid_img = make_grid(train_samples, nrow=8, normalize=True)
    grid_img = np.transpose(grid_img.numpy(), (1, 2, 0))
    plt.figure(figsize=(20, 4))
    plt.title('image samples')
    plt.imshow(grid_img, interpolation='nearest')
    plt.show()


def plot_metrics(metrics, outdir):
    # Loss over epochs
    plt.figure(figsize=(20, 5))
    plt.title('loss')
    plt.xlabel('epochs')
    for subset, reduce_steps in [('train', False), ('test', True)]:
        loss_data, n_epochs = plots.epoch_loss(subset, reduce_steps)
        if reduce_steps:
            loss_data = np.insert(loss_data, 0, loss_data[0])
        x = np.linspace(0, n_epochs, num=len(loss_data))
        plt.plot(x, loss_data, label=f'{subset} loss')
    plt.legend()
    plt.savefig(f'{outdir}/loss.jpg')

    # Accuracy (instance-balanced) over epochs
    plt.figure(figsize=(20, 5))
    plt.title('accuracy')
    plt.xlabel('epochs')
    for subset in ['train', 'test']:
        acc_data, n_epochs = plots.epoch_acc(subset)
        acc_data = np.insert(acc_data, 0, acc_data[0])
        x = np.linspace(0, n_epochs, num=len(acc_data))
        plt.plot(x, acc_data, label=f'{subset} acc')
    plt.legend()
    plt.savefig(f'{outdir}/epoch_acc.jpg')

    # Final class and instance based accuracies
    width = 0.2
    plt.figure(figsize=(20, 5))
    plt.xlabel('classes'), plt.title('class accuracy')
    for subset in ['train', 'test']:
        inst_acc, class_acc = plots.last_epoch_acc(subset)
        classes = np.arange(len(class_acc))
        plt.bar(classes - width / 2, class_acc, width,
                label=f'{subset} - {inst_acc:.3}/{class_acc.mean():.3} insta/class acc')
    plt.legend()
    plt.savefig(f'{outdir}/final_acc.jpg')

    plt.show()


def plot_similarity_hist(model, data_loader, outpath):
    """
    Assumes the model was trained with contrastive learning,
    which means that the data_loader should have more than 1 view
    """
    print('plotting similarities of features')
    inst_sims, class_sims, neg_sims = [], [], []
    model.eval(), model.cuda()
    feat_dim = None
    with torch.no_grad():
        for img_views, labels in data_loader:
            img_views = [imgs.cuda() for imgs in img_views]
            labels = labels.cuda()

            # Features
            feat_views = model.feats(img_views)
            project_views = model.project(feat_views)

            # All similarities
            all_sims = torch.matmul(project_views[0], project_views[1].T)

            # Masks to seperate types of similarities
            labels = labels.contiguous().view(-1, 1)
            eye = torch.eye(len(labels), dtype=torch.float32, device='cuda')
            mask = torch.eq(labels, labels.T).float()
            neg_mask = 1 - mask
            class_mask = mask - eye

            # Add similarities to respective groups
            inst_sims.extend(torch.diagonal(all_sims).cpu().tolist())
            class_sims.extend(torch.masked_select(all_sims, class_mask.bool()).cpu().tolist())
            neg_sims.extend(torch.masked_select(all_sims, neg_mask.bool()).cpu().tolist())

    # Plot similarities based on type
    plt.figure(figsize=(6, 6))
    plt.title(f'similarities of features')
    sim_data = [(inst_sims, 'inst'), (class_sims, 'class'), (neg_sims, 'neg')]
    for sims, label in sim_data:
        plt.hist(sims, density=True, label=label, alpha=0.3)
    plt.legend()

    # Save and show
    plt.savefig(outpath)
    plt.show()


from sklearn import manifold


def plot_tsne_similarity_types(model, data_loader, outpath):
    print('plotting t-SNE visualization of features by similarity type')
    model.eval(), model.cuda()

    aug_feats, class_feats, neg_feats = [], [], []
    with torch.no_grad():
        img_views, labels = next(iter(data_loader))
        nviews = len(img_views)
        bsz = len(labels)
        img_views = [imgs.cuda() for imgs in img_views]
        feat_views = model.feats(img_views)
        feat_views = [feats.cpu().numpy() for feats in feat_views]

        root_feat = feat_views[0][0]
        root_class = labels[0]

        # Class and negative similarities
        for i in range(1, nviews):
            # Instance similarities (augmented)
            aug_feats.append(feat_views[i][0])
            for j in range(bsz):
                if labels[j] == root_class:
                    # Class similarities
                    class_feats.append(feat_views[i][j])
                else:
                    # Negative similarities
                    neg_feats.append(feat_views[i][j])

    neg_feats = neg_feats[:bsz]

    X = np.concatenate([[root_feat], aug_feats, class_feats, neg_feats])
    y = np.concatenate([['root'], ['inst'] * len(aug_feats),
                        ['class'] * len(class_feats),
                        ['neg'] * len(neg_feats)])
    tsne = manifold.TSNE()
    X_embed = tsne.fit_transform(X)

    unique_y = np.unique(y)
    plt.figure(figsize=(12, 12))
    feat_dim = feat_views[0].shape[1]
    plt.title(f't-SNE, feature dimension: {feat_dim}')
    for target in unique_y:
        X_target = X_embed[y == target]
        assert X_target.shape[1] == 2
        plt.scatter(X_target[:, 0], X_target[:, 1], label=f'{target}', marker='.')
    if len(unique_y) <= 10:
        plt.legend()

    # Save and show
    plt.savefig(outpath)
    plt.show()


def plot_tsne(model, data_loader, outpath):
    print('plotting t-SNE visualization of features')
    X, y = [], []
    model.eval(), model.cuda()
    feat_dim = None
    with torch.no_grad():
        for img_views, labels in data_loader:
            img_views = [imgs.cuda() for imgs in img_views]
            feat_views = model.feats(img_views)
            feats = feat_views[0]
            feat_dim = feats.shape[1]
            X.extend(feats.cpu().tolist())
            y.extend(labels.tolist())
    X, y = np.array(X), np.array(y)

    tsne = manifold.TSNE()
    X_embed = tsne.fit_transform(X)

    unique_y = np.unique(y)
    plt.figure(figsize=(6, 6))
    plt.title(f't-SNE, feature dimension: {feat_dim}')
    for target in unique_y:
        X_target = X_embed[y == target]
        assert X_target.shape[1] == 2
        plt.scatter(X_target[:, 0], X_target[:, 1], label=f'{target}', alpha=0.2)
    if len(unique_y) <= 10:
        plt.legend()

    # Save and show
    plt.savefig(outpath)
    plt.show()
