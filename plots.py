import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


def plot_tsne(args, strategy, model, ds_val):
    from sklearn import manifold

    @tf.function
    def get_feats(imgs):
        feats = model.feats(imgs)
        proj = model.project(feats)
        return feats, proj

    all_feats, all_proj, all_labels = [], [], []
    for imgs1, imgs2, labels in tqdm(ds_val, 'tsne'):
        feats, proj = strategy.run(get_feats, (imgs1,))

        if args.tpu:
            feats, proj, labels = feats.values, proj.values, labels.values
        else:
            feats, proj, labels = [feats], [proj], [labels]

        for f, p, l in zip(feats, proj, labels):
            all_feats.append(f.numpy())
            all_proj.append(p.numpy())
            all_labels.append(l.numpy())

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


def plot_img_samples(args, ds_train, ds_val):
    f, ax = plt.subplots(2, 8)
    f.set_size_inches(20, 6)
    for i, ds in enumerate([ds_train, ds_val]):
        imgs, _, _ = next(iter(ds))
        for j in range(8):
            ax[i, j].set_title('train' if i == 0 else 'val')
            ax[i, j].imshow(imgs[j])

    f.tight_layout()
    f.savefig(os.path.join(args.out, 'img-samples.jpg'))


def plot_hist_sims(args, strategy, model, ds_val):
    @tf.function
    def get_sims(imgs1, imgs2, labels):
        # Features and similarities
        feats1, feats2 = model.feats(imgs1), model.feats(imgs2)
        proj1, proj2 = model.project(feats1), model.project(feats2)
        sims = tf.matmul(feats1, tf.transpose(feats2))
        proj_sims = tf.matmul(proj1, tf.transpose(proj2))

        # Masks
        bsz = len(labels)
        labels = tf.expand_dims(labels, 1)
        inst_mask = tf.eye(bsz, dtype=tf.bool)
        class_mask = (labels == tf.transpose(labels))
        pos_mask = inst_mask | class_mask
        neg_mask = ~pos_mask

        # Similarity types
        neg_sims = tf.boolean_mask(sims, neg_mask)
        class_sims = tf.boolean_mask(sims, class_mask)
        inst_sims = tf.boolean_mask(sims, inst_mask)

        proj_neg_sims = tf.boolean_mask(proj_sims, neg_mask)
        proj_class_sims = tf.boolean_mask(proj_sims, class_mask)
        proj_inst_sims = tf.boolean_mask(proj_sims, inst_mask)

        return (neg_sims, class_sims, inst_sims), (proj_neg_sims, proj_class_sims, proj_inst_sims)

    neg_sims, class_sims, inst_sims = np.array([]), np.array([]), np.array([])
    proj_neg_sims, proj_class_sims, proj_inst_sims = np.array([]), np.array([]), np.array([])
    for imgs1, imgs2, labels in tqdm(ds_val, 'similarity histogram'):
        sims, proj_sims = strategy.run(get_sims, (imgs1, imgs2, labels))

        if args.tpu:
            sims = sims.values
            proj_sims = proj_sims.values
        else:
            sims = [sims]
            proj_sims = [proj_sims]

        # Similarity types
        for s in sims:
            neg_sims = np.append(neg_sims, s[0].numpy())
            class_sims = np.append(class_sims, s[1].numpy())
            inst_sims = np.append(inst_sims, s[2].numpy())

        # Projected similarity types
        for p in proj_sims:
            proj_neg_sims = np.append(proj_neg_sims, p[0].numpy())
            proj_class_sims = np.append(proj_class_sims, p[1].numpy())
            proj_inst_sims = np.append(proj_inst_sims, p[2].numpy())

    # Plot
    f, ax = plt.subplots(1, 2)
    f.set_size_inches(13, 5)
    ax[0].set_title('similarity types')
    ax[0].hist(neg_sims, label='neg', weights=np.ones_like(neg_sims) / len(neg_sims), alpha=0.5)
    ax[0].hist(class_sims, label='class', weights=np.ones_like(class_sims) / len(class_sims), alpha=0.5)
    ax[0].hist(inst_sims, label='inst', weights=np.ones_like(inst_sims) / len(inst_sims), alpha=0.5)
    ax[0].legend()

    ax[1].set_title('projected similarity types')
    ax[1].hist(proj_neg_sims, label='neg', weights=np.ones_like(proj_neg_sims) / len(proj_neg_sims), alpha=0.5)
    ax[1].hist(proj_class_sims, label='class', weights=np.ones_like(proj_class_sims) / len(proj_class_sims), alpha=0.5)
    ax[1].hist(proj_inst_sims, label='inst', weights=np.ones_like(proj_inst_sims) / len(proj_inst_sims), alpha=0.5)
    ax[1].legend()

    f.savefig(os.path.join(args.out, 'similarity-types.jpg'))


def plot_metrics(args, train_df, val_df):
    all_metrics = ['acc', 'ce-loss', 'con-loss']
    f, ax = plt.subplots(1, 3)
    f.set_size_inches(20, 5)

    for i, metric in enumerate(all_metrics):
        ax[i].set_title(metric)
        for df, split in [(train_df, 'train'), (val_df, 'val')]:
            nsteps, nepochs = len(df), df['epoch'].max()
            x = np.linspace(0, nepochs, nsteps)
            ax[i].plot(x, df[metric], label=split)
            ax[i].legend()

    f.savefig(os.path.join(args.out, 'metrics.jpg'))
