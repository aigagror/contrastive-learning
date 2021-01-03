import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, losses, metrics

import optim
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

        # L2 regularization
        regularizer = keras.regularizers.l2(args.l2_reg)
        for layer in self.layers:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # Load weights?
        if args.load:
            # Call to build weights, then load
            print(f'loaded previously saved model weights')
            self.load_weights('gs://aigagror/contrastive-learning/model')
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

    def supcon_step(self, imgs1, imgs2, labels, bsz, train):
        with tf.GradientTape(watch_accessed_variables=train) as tape:
            # Features
            feats1, feats2 = self.norm_feats(imgs1), self.norm_feats(imgs2)
            proj1, proj2 = self.norm_project(feats1), self.norm_project(feats2)

            # Contrast loss
            con_loss = supcon_loss(labels, proj1, proj2, partial=False)
            con_loss = tf.nn.compute_average_loss(con_loss, global_batch_size=bsz)

            pred_logits = self.classifier(tf.stop_gradient(feats1))

            # Classifer cross entropy
            unreduced_ce_loss = losses.sparse_categorical_crossentropy(labels, tf.cast(pred_logits, tf.float32),
                                                                       from_logits=True)
            ce_loss = tf.nn.compute_average_loss(unreduced_ce_loss, global_batch_size=bsz)

            # Total loss
            loss = con_loss + ce_loss

        if train:
            # Gradient descent
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer = optim.get_global_optimizer()
            optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Accuracy
        acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
        acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
        return acc, ce_loss, con_loss

    def ce_step(self, imgs, labels, bsz, train):
        with tf.GradientTape(watch_accessed_variables=train) as tape:
            pred_logits, _ = self(imgs)

            # Classifer cross entropy
            loss = losses.sparse_categorical_crossentropy(labels, tf.cast(pred_logits, tf.float32), from_logits=True)
            loss = tf.nn.compute_average_loss(loss, global_batch_size=bsz)

        if train:
            # Gradient descent
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer = optim.get_global_optimizer()
            optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        # Accuracy
        acc = metrics.sparse_categorical_accuracy(labels, pred_logits)
        acc = tf.nn.compute_average_loss(acc, global_batch_size=bsz)
        return acc, loss, 0.0

    @tf.function
    def supcon_train(self, imgs1, imgs2, labels, bsz):
        return self.supcon_step(imgs1, imgs2, labels, bsz, train=True)

    @tf.function
    def supcon_val(self, imgs1, imgs2, labels, bsz):
        return self.supcon_step(imgs1, imgs2, labels, bsz, train=False)

    @tf.function
    def ce_train(self, imgs, labels, bsz):
        return self.ce_step(imgs, labels, bsz, train=True)

    @tf.function
    def ce_val(self, imgs, labels, bsz):
        return self.ce_step(imgs, labels, bsz, train=False)
