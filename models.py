import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, losses, metrics

from losses import supcon_loss


class ContrastModel(keras.Model):
    def __init__(self, args, nclass):
        super().__init__()
        if args.cnn == 'simple':
            self.cnn = keras.Sequential([
                layers.Conv2D(128, 3),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(128, 3),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.Conv2D(128, 3)
            ])
            self.preprocess = lambda img: img / 127.5 - 1
        else:
            assert args.cnn == 'resnet50v2'
            self.cnn = applications.ResNet50V2(weights=None, include_top=False)
            self.preprocess = applications.resnet_v2.preprocess_input

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.projection = layers.Dense(128, name='projection')
        self.classifier = layers.Dense(nclass, name='classifier')

        if args.load:
            print(f'loaded previously saved model weights')
            self.load_weights(os.path.join(args.out, 'model'))
        else:
            print(f'starting with new model weights')

    def norm_feats(self, img):
        x = self.preprocess(img * 255)
        x = self.cnn(x)
        x = self.avg_pool(x)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def norm_project(self, feats):
        x = self.projection(feats)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def call(self, img):
        feats = self.norm_feats(img)
        proj = self.norm_project(feats)
        return self.classifier(feats), proj

    @tf.function
    def train_step(self, imgs1, imgs2, labels, bsz, supcon):
        with tf.GradientTape() as tape:
            if False:
                # Features
                feats1, feats2 = self.norm_feats(imgs1), self.norm_feats(imgs2)
                proj1, proj2 = self.norm_project(feats1), self.norm_project(feats2)

                # Contrast
                con_loss = supcon_loss(labels, proj1, proj2, partial=False)
                con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

                pred_logits = self.classifier(tf.stop_gradient(feats1))
            else:
                con_loss = tf.constant(0, bsz.dtype)
                pred_logits, _ = self(imgs1)

            # Classifer cross entropy
            tf.debugging.assert_type(pred_logits, tf.uint8)
            tf.debugging.assert_less_equal(tf.reduce_max(labels), tf.constant(999, tf.int64))
            ce_loss = losses.sparse_categorical_crossentropy(labels, pred_logits, from_logits=True)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=bsz)
            loss = con_loss + ce_loss

            # Gradient descent
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Accuracy
        acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
        acc = tf.cast(acc, bsz.dtype)
        acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
        return acc, ce_loss, con_loss
