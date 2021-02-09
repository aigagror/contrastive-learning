import logging
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_tsne(args, strategy, model, ds_val):
    logging.info('plotting tsne of features')
    from sklearn import manifold

    outputs = [model.get_layer(name=name).output for name in ['feats', 'proj_feats']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def get_feats(inputs):
        feats, proj = feat_model(inputs)
        return feats, proj, inputs['label']

    all_feats, all_proj, all_labels = [], [], []
    for inputs in strategy.experimental_distribute_dataset(ds_val):
        feats, proj, labels = strategy.run(get_feats, [inputs])

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
    f, ax = plt.subplots(1, 2)
    f.set_size_inches(13, 5)
    ax[0].set_title('feats')
    ax[1].set_title('projected features')
    for c in classes:
        class_feats_embed = feats_embed[all_labels == c]
        class_proj_embed = proj_embed[all_labels == c]

        ax[0].scatter(class_feats_embed[:, 0], class_feats_embed[:, 1], label=f'{c}')
        ax[1].scatter(class_proj_embed[:, 0], class_proj_embed[:, 1], label=f'{c}')

    f.savefig(os.path.join('out/', 'tsne.jpg'))
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


def plot_hist_sims(args, strategy, model, ds_val):
    logging.info('plotting similarities histograms')

    outputs = [model.get_layer(name=name).output for name in ['feats', 'feats2', 'proj_feats', 'proj_feats2']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def get_all_feats(inputs):
        feats1, feats2, proj1, proj2 = feat_model(inputs)
        labels = inputs['label']
        return labels, feats1, feats2, proj1, proj2

    neg_sims, class_sims, inst_sims = np.array([]), np.array([]), np.array([])
    proj_neg_sims, proj_class_sims, proj_inst_sims = np.array([]), np.array([]), np.array([])

    for inputs in strategy.experimental_distribute_dataset(ds_val):
        labels, feats1, feats2, proj1, proj2 = strategy.run(get_all_feats, [inputs])

        # All gather
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

    f.savefig(os.path.join('out/', 'similarity-types.jpg'))
    logging.info("plotted similarity histograms to 'out/'")
