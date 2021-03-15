import logging
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def extract_sim_types(labels, feats1, feats2, proj1, proj2):
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


def get_sims(strategy, model, ds_val):
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

        sims, proj_sims = extract_sim_types(labels, feats1, feats2, proj1, proj2)

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


def log_sim_moments(sim_dict, proj_sim_dict):
    sim_moment_df, proj_sim_moment_df = pd.DataFrame(), pd.DataFrame()
    for key in ['neg', 'class', 'inst']:
        mean, std = np.mean(sim_dict[key]), np.std(sim_dict[key])
        sim_moment_df = sim_moment_df.append({'type': key, 'mean': mean, 'std': std}, ignore_index=True)

        mean, std = np.mean(proj_sim_dict[key]), np.std(proj_sim_dict[key])
        proj_sim_moment_df = proj_sim_moment_df.append({'type': key, 'mean': mean, 'std': std}, ignore_index=True)
    sim_moment_df.to_csv('out/sim-moments.csv')
    proj_sim_moment_df.to_csv('out/proj-sim-moments.csv')
    logging.info("saved similarity moments to 'out/'")


def plot_hist_sims(args, strategy, model, ds_val):
    logging.info('plotting similarities histograms')

    sim_dict, proj_sim_dict = get_sims(strategy, model, ds_val)

    # Plot similarity histograms
    plot_sim_hist(sim_dict, proj_sim_dict)

    # Log similarity moments
    log_sim_moments(sim_dict, proj_sim_dict)


def get_feats(strategy, model, ds_val):
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
    return all_feats, all_labels, all_proj


def plot_tsne(args, strategy, model, ds_val):
    logging.info('plotting tsne of features')

    all_feats, all_labels, all_proj = get_feats(strategy, model, ds_val)

    from sklearn import manifold
    feats_embed = manifold.TSNE().fit_transform(all_feats)
    proj_embed = manifold.TSNE().fit_transform(all_proj)

    classes = np.unique(all_labels)

    plt.figure()
    plt.title('TSNE features')
    for c in classes:
        class_feats_embed = feats_embed[all_labels == c]
        plt.scatter(class_feats_embed[:, 0], class_feats_embed[:, 1], label=f'{c}', alpha=0.1)
    plt.savefig(os.path.join('out/', 'tsne.pdf'))

    plt.figure()
    plt.title('TSNE projected features')
    for c in classes:
        class_proj_embed = proj_embed[all_labels == c]
        plt.scatter(class_proj_embed[:, 0], class_proj_embed[:, 1], label=f'{c}', alpha=0.1)
    plt.savefig(os.path.join('out/', 'proj-tsne.pdf'))

    logging.info("plotted tsne to 'out/'")
