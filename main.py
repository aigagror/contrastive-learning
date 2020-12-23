# -*- coding: utf-8 -*-
"""contrastive learning.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/186x4ArfFLbhkROefChnTbvdBzWXXMGau

## Setup
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, losses, metrics
import tensorflow.keras.mixed_precision.experimental as mixed_precision
import tensorflow_datasets as tfds

import os
import numpy as np
from tqdm.auto import tqdm

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int)
parser.add_argument('--bsz', type=int)

parser.add_argument('--lr', type=float)
parser.add_argument('--load', action='store_true')

parser.add_argument('--method', choices=['ce', 'supcon', 'supcon-pce'])

parser.add_argument('--out', type=str, default='./out/')

"""## Data"""

AUTOTUNE = -1

class Augment(layers.Layer):
  def __init__(self, imsize, rand_crop, rand_flip, rand_jitter, rand_gray):
    super().__init__(name='image-augmentation')
    self.imsize = imsize
    self.rand_crop = rand_crop
    self.rand_flip = rand_flip
    self.rand_jitter = rand_jitter
    self.rand_gray = rand_gray

  @tf.function
  def call(self, image):
    print('traced image augmentation')

    # Convert to float
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop
    if self.rand_crop:
      rand_scale = tf.random.uniform([], 1, 1.5)
      rand_size = tf.round(rand_scale * self.imsize)
      image = tf.image.resize(image, [rand_size, rand_size])
      image = tf.image.random_crop(image, [self.imsize, self.imsize, 3])
    else:
      image = tf.image.resize(image, [self.imsize, self.imsize])
    
    # Random flip
    if self.rand_flip:
      image = tf.image.random_flip_left_right(image)
    
    # Color Jitter
    if self.rand_jitter and tf.random.uniform([]) < 0.8:
      image = tf.image.random_brightness(image, 0.4)
      image = tf.image.random_contrast(image, 0.6, 1.4)
      image = tf.image.random_saturation(image, 0.6, 1.4)
      image = tf.image.random_hue(image, 0.1)
    
    # Gray scale
    if self.rand_gray and tf.random.uniform([]) < 0.2:
      image = tf.image.rgb_to_grayscale(image)
      image = tf.tile(image, [1, 1, 3])
    
    # Clip
    image = tf.clip_by_value(image, 0, 1)
    
    return image

def load_datasets(args, strategy):
  (ds_train, ds_test), ds_info = tfds.load('cifar10', split=['train', 'test'],
                                        as_supervised=True, shuffle_files=True,  
                                        with_info=True)
  
  augment = Augment(imsize=32, rand_crop=False, rand_flip=True, 
                    rand_jitter=False, rand_gray=True)
  def train_map(imgs, labels):
    return augment(imgs), labels

  ds_train = (
      ds_train
      .cache()
      .map(train_map, num_parallel_calls=AUTOTUNE)
      .shuffle(ds_info.splits['train'].num_examples)
      .batch(args.bsz, drop_remainder=True)
      .prefetch(AUTOTUNE)
  )
  ds_test = (
      ds_test
      .cache()
      .shuffle(ds_info.splits['test'].num_examples)
      .batch(args.bsz)
      .prefetch(AUTOTUNE)
  )

  ds_train = strategy.experimental_distribute_dataset(ds_train)
  ds_test = strategy.experimental_distribute_dataset(ds_test)

  return ds_train, ds_test

"""## Loss"""

@tf.function
def supcon_loss(labels, feats1, feats2, partial):  
  tf.debugging.assert_all_finite(feats1, 'feats1 not finite')
  tf.debugging.assert_all_finite(feats2, 'feats2 not finite')
  bsz = len(labels)
  labels = tf.expand_dims(labels, 1)

  # Masks
  inst_mask = tf.eye(bsz)
  class_mask = tf.cast(labels == tf.transpose(labels), tf.float32)
  class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

  # Similarities
  sims = tf.matmul(feats1, tf.transpose(feats2))
  tf.debugging.assert_all_finite(sims, 'similarities not finite')
  tf.debugging.assert_less_equal(sims, tf.ones_like(sims) + 1e-2, 
                                 'similarities not less than or equal to 1')

  if partial:
    print('traced supcon loss - partial cross entropy')
    # Partial cross entropy
    pos_mask = tf.maximum(inst_mask, class_mask)
    neg_mask = 1 - pos_mask

    exp = tf.math.exp(sims * 10)
    neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
    log_prob = sims - tf.math.log(neg_sum_exp + exp)

    # Class positive pairs log prob (contains instance positive pairs too)
    class_log_prob = class_mask * log_prob
    class_log_prob = tf.math.reduce_sum(class_log_prob / class_sum, axis=1)

    loss = -class_log_prob
  else:
    print('traced supcon loss - cross entropy')
    # Cross entropy
    loss = losses.categorical_crossentropy(class_mask / class_sum, sims * 10, 
                                           from_logits=True)    
  return loss

"""## Model"""

class ContrastModel(keras.Model):
  def __init__(self, args):
    super().__init__()
    
    self.cnn = applications.ResNet50V2(weights=None, include_top=False) 
    self.avg_pool = layers.GlobalAveragePooling2D()
    self.proj_w = layers.Dense(128, name='projection')
    self.classifier = layers.Dense(10, name='classifier')

  def feats(self, img):
    x = applications.resnet_v2.preprocess_input(img)
    x = self.cnn(img)
    x = self.avg_pool(x)
    x, _ = tf.linalg.normalize(x, axis=-1)
    return x

  def project(self, feats):
    x = self.proj_w(feats)
    x, _ = tf.linalg.normalize(x, axis=-1)
    return x
  
  def call(self, img):
    print('traced model call')
    feats = self.feats(img)
    proj = self.project(feats)
    return self.classifier(feats), proj

  @tf.function
  def train_step(self, method, bsz, imgs1, labels):
    with tf.GradientTape() as tape:
      if method.startswith('supcon'):
        print(f'traced model train step - supcon')
        partial = method.endswith('pce')

        # Features
        feats1 = self.feats(imgs1)
        proj1 = self.project(feats1)

        # Contrast
        con_loss = supcon_loss(labels, proj1, proj1, partial)
        con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

        pred_logits = self.classifier(tf.stop_gradient(feats1))
      elif method == 'ce':
        print(f'traced model train step - cross entropy')
        con_loss = 0
        pred_logits, _ = self(imgs1)
      else:
        raise Exception(f'unknown train method {method}')

      # Classifer cross entropy
      class_loss = losses.sparse_categorical_crossentropy(labels, pred_logits, 
                                                          from_logits=True)
      class_loss = tf.nn.compute_average_loss(class_loss, global_batch_size=bsz)
      loss = con_loss + class_loss
      scaled_loss = self.optimizer.get_scaled_loss(loss)

    # Gradient descent
    scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
    gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

    # Accuracy
    acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
    acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
    return loss, acc

"""## Train"""

def epoch_train(args, model, strategy, ds_train):
  accs, losses = [], []
  pbar = tqdm(ds_train, leave=False, mininterval=1)
  for imgs1, labels in pbar:
    # Train step
    loss, acc = strategy.run(model.train_step, 
                             args=(args.method, args.bsz, imgs1, labels))
    loss = strategy.reduce('SUM', loss, axis=None)
    acc = strategy.reduce('SUM', acc, axis=None)

    # Record
    losses.append(float(loss))
    accs.append(float(acc))
    pbar.set_postfix_str(f'{losses[-1]:.3} loss, {accs[-1]:.3} acc', 
                         refresh=False)
  return accs, losses

def train(args, model, strategy, ds_train, ds_test):
  all_accs, all_losses = [], []

  try:
    pbar = tqdm(range(args.epochs), 'epochs')
    for epoch in pbar:
      accs, losses = epoch_train(args, model, strategy, ds_train)
      model.save_weights(os.path.join(args.out, 'model'))
      all_accs.extend(accs)
      all_losses.extend(losses)
      pbar.set_postfix_str(f'{np.mean(losses):.3} loss, {np.mean(accs):.3} acc')
  except KeyboardInterrupt:
    print('keyboard interrupt caught. ending training early')

  return all_accs, all_losses

"""## Plot"""

import matplotlib.pyplot as plt

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

from sklearn import manifold

def plot_tsne(args, model, ds_test):
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

def plot_metrics(args, accs, losses):
  nsteps = len(accs)
  assert len(losses) == nsteps
  x = np.linspace(0, args.epochs, nsteps)
  f, ax = plt.subplots(1, 2)
  f.set_size_inches(13, 5)

  ax[0].set_title('accuracy')
  ax[0].set_xlabel('epochs')
  ax[0].plot(x, accs)

  ax[1].set_title('losses')
  ax[1].plot(x, losses)
  ax[1].set_xlabel('epochs')
  
  f.savefig(os.path.join(args.out, 'metrics.jpg'))
  plt.show()

"""## Main"""

def run(args):
  # Mixed precision
  policy = mixed_precision.Policy('mixed_float16')  
  mixed_precision.set_global_policy(policy)

  # Strategy
  gpus = tf.config.list_physical_devices('GPU')
  print(f'GPUs: {gpus}')
  if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
  else:
    strategy = tf.distribute.get_strategy() 

  # Data
  ds_train, ds_test = load_datasets(args, strategy)
  plot_img_samples(args,ds_train, ds_test)

  # Model and optimizer
  with strategy.scope():
    model = ContrastModel(args)
    opt = keras.optimizers.SGD(args.lr, momentum=0.9)
    model.optimizer = mixed_precision.LossScaleOptimizer(opt)
  if args.load:
    model.load_weights(os.path.join(args.out, 'model'))

  # Train
  accs, losses = train(args, model, strategy, ds_train, ds_test)

  # Plot
  plot_metrics(args, accs, losses)
  plot_tsne(args, model, ds_test)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    run(args)

