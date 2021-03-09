import logging
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_tsne(args, strategy, model, ds_val, max_iter=None):
    logging.info('plotting tsne of features')
    from sklearn import manifold

    outputs = [model.get_layer(name=name).output for name in ['feats', 'proj_feats']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def get_feats(inputs, targets):
        feats, proj = feat_model(inputs)
        return feats, proj, targets['label']

    all_feats, all_proj, all_labels = [], [], []
    for i, (inputs, targets) in enumerate(ds_val):
        if max_iter is not None and i >= max_iter:
            break
        feats, proj, labels = strategy.run(get_feats, [inputs, targets])

        feats = strategy.gather(feats, axis=0)
        proj = strategy.gather(proj, axis=0)
        labels = strategy.gather(labels, axis=0)

        all_feats.append(feats.numpy())
        all_proj.append(proj.numpy())
        all_labels.append(labels.numpy())

    all_feats = np.concatenate(all_feats)
    all_proj = np.concatenate(all_proj)
    all_labels = np.concatenate(all_labels)

    feats_embed = manifold.TSNE().fit_transform(all_feats)
    proj_embed = manifold.TSNE().fit_transform(all_proj)

    classes = np.unique(all_labels)

    plt.figure()
    plt.title('TSNE features')
    for c in classes:
        class_feats_embed = feats_embed[all_labels == c]
        plt.scatter(class_feats_embed[:, 0], class_feats_embed[:, 1], label=f'{c}', alpha=0.1)
    plt.savefig(os.path.join('out/', 'proj-tsne.pdf'))

    plt.figure()
    plt.title('TSNE projected features')
    for c in classes:
        class_proj_embed = proj_embed[all_labels == c]
        plt.scatter(class_proj_embed[:, 0], class_proj_embed[:, 1], label=f'{c}', alpha=0.1)
    plt.savefig(os.path.join('out/', 'proj-tsne.pdf'))

    logging.info("plotted tsne to 'out/'")


def get_all_sims(labels, feats1, feats2, proj1, proj2):
    # Similarities
    sims = tf.matmul(feats1, tf.transpose(feats2))
    proj_sims = tf.matmul(proj1, tf.transpose(proj2))

    # Masks
    bsz = len(labels)
    labels = tf.expand_dims(labels, 1)
    tf.debugging.assert_shapes([
        (labels, [None, 1])
    ])
    inst_mask = tf.eye(bsz, dtype=tf.bool)
    class_mask = (labels == tf.transpose(labels))
    class_mask = tf.linalg.set_diag(class_mask, tf.zeros(bsz, tf.bool))
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


def plot_hist_sims(args, strategy, model, ds_val, max_iter=None):
    logging.info('plotting similarities histograms')

    outputs = [model.get_layer(name=name).output for name in ['feats', 'feats2', 'proj_feats', 'proj_feats2']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def get_all_feats(inputs, targets):
        feats1, feats2, proj1, proj2 = feat_model(inputs)
        labels = targets['label']
        return labels, feats1, feats2, proj1, proj2

    neg_sims, class_sims, inst_sims = np.array([]), np.array([]), np.array([])
    proj_neg_sims, proj_class_sims, proj_inst_sims = np.array([]), np.array([]), np.array([])

    for i, (inputs, targets) in enumerate(ds_val):
        if max_iter is not None and i >= max_iter:
            break

        labels, feats1, feats2, proj1, proj2 = strategy.run(get_all_feats, [inputs, targets])

        # All gather
        tf.print(feats1)
        feats1 = strategy.gather(feats1, axis=0)
        feats2 = strategy.gather(feats2, axis=0)
        proj1 = strategy.gather(proj1, axis=0)
        proj2 = strategy.gather(proj2, axis=0)
        labels = strategy.gather(labels, axis=0)

        sims, proj_sims = get_all_sims(labels, feats1, feats2, proj1, proj2)

        # Similarity types
        neg_sims = np.append(neg_sims, sims[0].numpy())
        class_sims = np.append(class_sims, sims[1].numpy())
        inst_sims = np.append(inst_sims, sims[2].numpy())

        # Projected similarity types
        proj_neg_sims = np.append(proj_neg_sims, proj_sims[0].numpy())
        proj_class_sims = np.append(proj_class_sims, proj_sims[1].numpy())
        proj_inst_sims = np.append(proj_inst_sims, proj_sims[2].numpy())

    # Plot
    plt.figure()
    plt.title('similarities')
    plt.hist(neg_sims, label='neg', weights=np.ones_like(neg_sims) / len(neg_sims), alpha=0.5)
    plt.hist(class_sims, label='class', weights=np.ones_like(class_sims) / len(class_sims), alpha=0.5)
    plt.hist(inst_sims, label='inst', weights=np.ones_like(inst_sims) / len(inst_sims), alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join('out/', 'sims.pdf'))

    plt.figure()
    plt.title('projected similarities')
    plt.hist(proj_neg_sims, label='neg', weights=np.ones_like(proj_neg_sims) / len(proj_neg_sims), alpha=0.5)
    plt.hist(proj_class_sims, label='class', weights=np.ones_like(proj_class_sims) / len(proj_class_sims), alpha=0.5)
    plt.hist(proj_inst_sims, label='inst', weights=np.ones_like(proj_inst_sims) / len(proj_inst_sims), alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join('out/', 'proj-sims.pdf'))

    logging.info("plotted similarity histograms to 'out/'")
