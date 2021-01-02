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
            self.preprocess = lambda img: 2 * img - 1
        else:
            assert args.cnn == 'resnet50v2'
            self.cnn = applications.ResNet50V2(weights=None, include_top=False)
            self.preprocess = applications.resnet_v2.preprocess_input

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.proj_w = layers.Dense(128, name='projection')
        self.classifier = layers.Dense(nclass, name='classifier')

        if args.load:
            print(f'loaded previously saved model weights')
            self.load_weights(os.path.join(args.out, 'model'))
        else:
            print(f'starting with new model weights')

    def feats(self, img):
        x = self.preprocess(img * 255)
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
    def train_step(self, imgs1, imgs2, labels, bsz, supcon):
        with tf.GradientTape() as tape:
            if supcon:
                partial = self.method.endswith('pce')

                # Features
                feats1, feats2 = self.feats(imgs1), self.feats(imgs2)
                proj1, proj2 = self.project(feats1), self.project(feats2)

                # Contrast
                con_loss = supcon_loss(labels, proj1, proj2, partial)
                con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

                pred_logits = self.classifier(tf.stop_gradient(feats1))
            else:
                con_loss = 0
                pred_logits, _ = self(imgs1)

            # Classifer cross entropy
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
