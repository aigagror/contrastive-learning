import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, losses, metrics

from losses import supcon_loss


class ContrastModel(keras.Model):
    def __init__(self, args):
        super().__init__()

        self.cnn = applications.ResNet50V2(weights=None, include_top=False)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.proj_w = layers.Dense(128, name='projection')
        self.classifier = layers.Dense(10, name='classifier')

        if args.load:
            print(f'loaded previously saved model weights')
            self.load_weights(os.path.join(args.out, 'model'))
        else:
            print(f'starting with new model weights')

    def feats(self, img):
        x = img * 2 - 1
        x = self.cnn(x)
        x = self.avg_pool(x)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def project(self, feats):
        x = self.proj_w(feats)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def call(self, img):
        feats = self.feats(img)
        proj = self.project(feats)
        return self.classifier(feats), proj

    @tf.function
    def train_step(self, method, bsz, imgs1, imgs2, labels, optimize):
        with tf.GradientTape(watch_accessed_variables=optimize) as tape:
            if method.startswith('supcon'):
                partial = method.endswith('pce')

                # Features
                feats1, feats2 = self.feats(imgs1), self.feats(imgs2)
                proj1, proj2 = self.project(feats1), self.project(feats2)

                # Contrast
                con_loss = supcon_loss(labels, proj1, tf.stop_gradient(proj2), partial)
                con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

                pred_logits = self.classifier(tf.stop_gradient(feats1))
            elif method == 'ce':
                con_loss = 0
                pred_logits, _ = self(imgs1)
            else:
                raise Exception(f'unknown train method {method}')

            # Classifer cross entropy
            ce_loss = losses.sparse_categorical_crossentropy(labels, pred_logits,
                                                             from_logits=True)
            ce_loss = tf.nn.compute_average_loss(ce_loss, global_batch_size=bsz)
            loss = con_loss + ce_loss
            scaled_loss = self.optimizer.get_scaled_loss(loss)

        # Gradient descent
        if optimize:
            scaled_gradients = tape.gradient(scaled_loss, self.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_gradients)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Accuracy
        acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
        acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
        return acc, con_loss, ce_loss
