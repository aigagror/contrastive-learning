import os

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging
from matplotlib import pyplot as plt
from sklearn import manifold

from data import augmentations


def _extract_sim_types(labels, feats1, feats2, proj1, proj2):
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


def _tf_get_sims(strategy, model, ds_val):
    outputs = [model.get_layer(name=name).output for name in ['feats', 'feats2', 'proj_feats', 'proj_feats2']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def run_model(inputs, targets):
        feats1, feats2, proj1, proj2 = feat_model(inputs)
        labels = targets['label']
        return labels, feats1, feats2, proj1, proj2

    neg_sims, class_sims, inst_sims = np.array([]), np.array([]), np.array([])
    proj_neg_sims, proj_class_sims, proj_inst_sims = np.array([]), np.array([]), np.array([])
    for i, (inputs, targets) in enumerate(ds_val):
        labels, feats1, feats2, proj1, proj2 = strategy.run(run_model, [inputs, targets])

        # All gather
        feats1 = strategy.gather(feats1, axis=0)
        feats2 = strategy.gather(feats2, axis=0)
        proj1 = strategy.gather(proj1, axis=0)
        proj2 = strategy.gather(proj2, axis=0)
        labels = strategy.gather(labels, axis=0)

        sims, proj_sims = _extract_sim_types(labels, feats1, feats2, proj1, proj2)

        # Similarity types
        neg_sims = np.append(neg_sims, sims[0].numpy())
        class_sims = np.append(class_sims, sims[1].numpy())
        inst_sims = np.append(inst_sims, sims[2].numpy())

        # Projected similarity types
        proj_neg_sims = np.append(proj_neg_sims, proj_sims[0].numpy())
        proj_class_sims = np.append(proj_class_sims, proj_sims[1].numpy())
        proj_inst_sims = np.append(proj_inst_sims, proj_sims[2].numpy())
    sim_dict = {'neg': neg_sims, 'class': class_sims, 'inst': inst_sims}
    proj_sim_dict = {'neg': proj_neg_sims, 'class': proj_class_sims, 'inst': proj_inst_sims}
    return sim_dict, proj_sim_dict


def _tf_get_feats_and_labels(strategy, model, ds_val):
    outputs = [model.get_layer(name=name).output for name in ['feats', 'proj_feats']]
    feat_model = tf.keras.Model(model.input, outputs)

    @tf.function
    def run_model(inputs, targets):
        feats, proj = feat_model(inputs)
        return feats, proj, targets['label']

    all_feats, all_proj, all_labels = [], [], []
    for i, (inputs, targets) in enumerate(ds_val):
        feats, proj, labels = strategy.run(run_model, [inputs, targets])

        feats = strategy.gather(feats, axis=0)
        proj = strategy.gather(proj, axis=0)
        labels = strategy.gather(labels, axis=0)

        all_feats.append(feats.numpy())
        all_proj.append(proj.numpy())
        all_labels.append(labels.numpy())

    all_feats = np.concatenate(all_feats)
    all_proj = np.concatenate(all_proj)
    all_labels = np.concatenate(all_labels)
    return all_feats, all_proj, all_labels


def _log_sim_moments(sim_dict, proj_sim_dict):
    sim_moment_df, proj_sim_moment_df = pd.DataFrame(), pd.DataFrame()
    for key in ['neg', 'class', 'inst']:
        mean, std = np.mean(sim_dict[key]), np.std(sim_dict[key])
        sim_moment_df = sim_moment_df.append({'type': key, 'mean': mean, 'std': std}, ignore_index=True)

        mean, std = np.mean(proj_sim_dict[key]), np.std(proj_sim_dict[key])
        proj_sim_moment_df = proj_sim_moment_df.append({'type': key, 'mean': mean, 'std': std}, ignore_index=True)
    sim_moment_df.to_csv('out/sim-moments.csv')
    proj_sim_moment_df.to_csv('out/proj-sim-moments.csv')
    logging.info("saved similarity moments to 'out/'")


def _scatter_tsne_label(filepath, feats_embed, all_labels, title=None, legend=False):
    plt.figure()
    plt.title(title)
    classes = np.unique(all_labels)
    for c in classes:
        class_feats_embed = feats_embed[all_labels == c]
        plt.scatter(class_feats_embed[:, 0], class_feats_embed[:, 1], label=c, alpha=0.2)
    if legend:
        plt.legend()
    plt.savefig(filepath)
    return classes


def plot_sim_hist(sim_dict, proj_sim_dict):
    plt.figure()
    plt.title('similarities')
    for key in ['neg', 'class', 'inst']:
        plt.hist(sim_dict[key], label=key, weights=np.ones_like(sim_dict[key]) / len(sim_dict[key]), alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join('out/', 'sims.pdf'))
    plt.figure()
    plt.title('projected similarities')
    for key in ['neg', 'class', 'inst']:
        plt.hist(proj_sim_dict[key], label=key, weights=np.ones_like(proj_sim_dict[key]) / len(proj_sim_dict[key]),
                 alpha=0.5)
    plt.legend()
    plt.savefig(os.path.join('out/', 'proj-sims.pdf'))
    logging.info("plotted similarity histograms to 'out/'")


def plot_hist_sims(args, strategy, model, ds_val):
    logging.info('plotting similarities histograms')

    sim_dict, proj_sim_dict = _tf_get_sims(strategy, model, ds_val)

    # Plot similarity histograms
    plot_sim_hist(sim_dict, proj_sim_dict)

    # Log similarity moments
    _log_sim_moments(sim_dict, proj_sim_dict)


def plot_tsne(args, strategy, model, ds_val):
    logging.info('plotting tsne of features')

    all_feats, all_proj, all_labels = _tf_get_feats_and_labels(strategy, model, ds_val)

    feats_embed = manifold.TSNE().fit_transform(all_feats)
    proj_embed = manifold.TSNE().fit_transform(all_proj)

    _scatter_tsne_label(os.path.join('out/', 'tsne.pdf'), feats_embed, all_labels,
                        title='TSNE features')

    _scatter_tsne_label(os.path.join('out/', 'proj-tsne.pdf'), proj_embed, all_labels,
                        title='TSNE projected features')

    logging.info("plotted tsne to 'out/'")


def plot_instance_tsne(args, model, local_ds_val):
    logging.info('plotting instance tsne of features')

    # Anchor image
    ds_anchor = local_ds_val.unbatch().shuffle(10).take(1)
    images, targets = next(iter(ds_anchor))
    anchor_image = images['image']
    anchor_class = targets['label']
    logging.info(f'class: {anchor_class}')

    # Number of samples per category
    n_samples = 1024

    # Instance augmentations
    autoaugment = augmentations.AutoAugment('v0' if tf.shape(anchor_image)[-1] == 3 else 'gray')
    augment_fn = lambda x: autoaugment.distort(tf.image.random_flip_left_right(x))
    augmented_images = []
    for _ in range(n_samples):
        augmented_images.append(augment_fn(anchor_image))

    # Class and negatives
    ds_class = local_ds_val.unbatch().filter(lambda x, y: y['label'] == anchor_class)
    ds_neg = local_ds_val.unbatch().filter(lambda x, y: y['label'] != anchor_class)

    class_samples = list(ds_class.take(n_samples).map(lambda x, y: x['image']).as_numpy_iterator())
    neg_samples = list(ds_neg.take(n_samples).map(lambda x, y: x['image']).as_numpy_iterator())

    all_images = [anchor_image] + augmented_images + class_samples + neg_samples
    all_images = tf.stack(all_images)
    all_labels = np.array(['anchor']
                          + ['instance'] * len(augmented_images)
                          + ['class'] * len(class_samples)
                          + ['negative'] * len(neg_samples))

    # Compute features
    feat_model = tf.keras.Model(model.input[0], model.get_layer(name='feats').output)
    proj_feat_model = tf.keras.Model(model.input[0], model.get_layer(name='proj_feats').output)

    feats = feat_model(all_images)
    proj_feats = proj_feat_model(all_images)

    # Compute TSNE
    feats_embed = manifold.TSNE().fit_transform(feats)
    proj_embed = manifold.TSNE().fit_transform(proj_feats)

    # Plot
    _scatter_tsne_label(os.path.join('out/', 'inst-tsne.pdf'), feats_embed, all_labels,
                        title='Instance TSNE features', legend=True)

    _scatter_tsne_label(os.path.join('out/', 'inst-proj-tsne.pdf'), proj_embed, all_labels,
                        title='Instance TSNE projected features', legend=True)

    logging.info("plotted instance tsne to 'out/'")
