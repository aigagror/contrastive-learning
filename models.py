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
        feats = self.feats(img)
        proj = self.project(feats)
        return self.classifier(feats), proj

    @tf.function
    def train_step(self, method, bsz, imgs1, labels):
        with tf.GradientTape() as tape:
            if method.startswith('supcon'):
                partial = method.endswith('pce')

                # Features
                feats1 = self.feats(imgs1)
                proj1 = self.project(feats1)

                # Contrast
                con_loss = supcon_loss(labels, proj1, tf.stop_gradient(proj1), partial)
                con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

                pred_logits = self.classifier(tf.stop_gradient(feats1))
            elif method == 'ce':
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

    @tf.function
    def test_step(self, bsz, imgs1, labels):
        imgs1 = tf.image.convert_image_dtype(imgs1, tf.float32)
        pred_logits, _ = self(imgs1)

        acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
        acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
        return acc
