import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_img_samples(args, ds_train, ds_test):
    f, ax = plt.subplots(2, 8)
    f.set_size_inches(20, 6)
    for i, ds in enumerate([ds_train, ds_test]):
        imgs = next(iter(ds))[0]
        for j in range(8):
            ax[i, j].set_title('train' if i == 0 else 'test')
            ax[i, j].imshow(imgs[j])

    f.tight_layout()
    f.savefig(os.path.join(args.out, 'img-samples.jpg'))
    plt.show()


def plot_tsne(args, model, ds_test):
    from sklearn import manifold

    all_feats, all_proj, all_labels = [], [], []
    for imgs, labels in ds_test:
        imgs = tf.image.convert_image_dtype(imgs, tf.float32)
        feats = model.feats(imgs)
        proj = model.project(feats)

        all_feats.append(feats.numpy())
        all_proj.append(proj.numpy())
        all_labels.append(labels.numpy())

    all_feats = np.concatenate(all_feats)
    all_proj = np.concatenate(all_proj)
    all_labels = np.concatenate(all_labels)

    feats_embed = manifold.TSNE().fit_transform(all_feats)
    proj_embed = manifold.TSNE().fit_transform(all_proj)

    classes = np.unique(all_labels)
    f, ax = plt.subplots(1, 2)
    f.set_size_inches(13, 5)
    ax[0].set_title('feats')
    ax[1].set_title('projected features')
    for c in classes:
        class_feats_embed = feats_embed[all_labels == c]
        class_proj_embed = proj_embed[all_labels == c]

        ax[0].scatter(class_feats_embed[:, 0], class_feats_embed[:, 1], label=f'{c}')
        ax[1].scatter(class_proj_embed[:, 0], class_proj_embed[:, 1], label=f'{c}')

    f.savefig(os.path.join(args.out, 'tsne.jpg'))
    plt.show()


def plot_metrics(args, metrics):
    f, ax = plt.subplots(1, len(metrics))
    f.set_size_inches(15, 5)

    names = ['test accs', 'train accs', 'train losses']
    for i, (y, name) in enumerate(zip(metrics, names)):
        y = np.array(y)
        x = np.linspace(0, len(y), y.size)
        ax[i].set_title(name)
        ax[i].set_xlabel('epochs')
        ax[i].plot(x, y.flatten())

    f.savefig(os.path.join(args.out, 'metrics.jpg'))
    plt.show()