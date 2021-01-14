import os

import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import applications, layers, metrics, losses


class ContrastModel(keras.Model):
    def __init__(self, args, nclass):
        super().__init__()
        self.args = args

        self.cnn = applications.ResNet50(weights=None, include_top=False)

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
            self.load_weights(os.path.join(args.out, 'model'))
        else:
            print(f'starting with new model weights')

    def features(self, img):
        x = tf.cast(img, tf.float32) / 127.5 - 1
        x = self.cnn(x)
        x = self.avg_pool(x)
        if self.args.norm_feats:
            l2 = tf.stop_gradient(tf.linalg.norm(x, axis=-1, keepdims=True))
            x = x / l2
        return x

    def projection(self, feats):
        x = self.projection(feats)
        if self.args.norm_feats:
            l2 = tf.stop_gradient(tf.linalg.norm(x, axis=-1, keepdims=True))
            x = x / l2
        return x

    def call(self, input, **kwargs):
        if self.args.method == 'ce':
            feats = self.features(input['imgs'])
            pred_logits = self.classifier(feats)
        else:
            assert self.args.method.startswith('supcon')
            partial = self.args.method.endswith('-pce')

            feats = self.features(input['imgs'])
            proj_feats = self.projection(feats)

            feats2 = self.features(input['imgs2'])
            proj_feats2 = self.projection(feats2)

            supcon_loss = self.compute_supcon_loss(input['labels'], proj_feats, proj_feats2, partial)
            supcon_loss = tf.reduce_mean(supcon_loss)
            self.add_loss(supcon_loss)
            self.add_metric(supcon_loss, 'supcon')

            pred_logits = self.classifier(tf.stop_gradient(feats))

        # Cross entropy and accuracy
        ce_loss = losses.sparse_categorical_crossentropy(input['labels'], pred_logits, from_logits=True)
        ce_loss = tf.reduce_mean(ce_loss)
        acc = metrics.sparse_categorical_accuracy(input['labels'], pred_logits)
        acc = tf.reduce_mean(acc)
        self.add_loss(ce_loss)
        self.add_metric(ce_loss, 'ce')
        self.add_metric(acc, 'acc')

        # Prediction
        pred = tf.argmax(pred_logits, axis=1)
        return pred

    def compute_supcon_loss(self, labels, feats1, feats2, partial):
        bsz = len(labels)
        tf.debugging.assert_shapes([(labels, [None, 1])])
        dtype = feats1.dtype

        # Masks
        inst_mask = tf.eye(bsz, dtype=dtype)
        class_mask = tf.cast(labels == tf.transpose(labels), dtype)
        class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

        # Similarities
        sims = tf.matmul(feats1, tf.transpose(feats2))

        if partial:
            # Cross entropy on instance similarities
            inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims * 10)

            # Partial cross entropy on class similarities
            pos_mask = tf.maximum(inst_mask, class_mask)
            neg_mask = 1 - pos_mask

            exp = tf.math.exp(sims * 10)
            neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
            log_prob = sims - tf.math.log(neg_sum_exp + exp)

            # Class positive pairs log prob (contains instance positive pairs too)
            class_log_prob = class_mask * log_prob
            class_log_prob = tf.math.reduce_sum(class_log_prob / class_sum, axis=1)
            class_loss = -class_log_prob

            # Combine instance loss and class loss
            loss = inst_loss + class_loss
            self.add_metric(tf.reduce_mean(inst_loss), 'inst-ce')
            self.add_metric(tf.reduce_mean(class_loss), 'class-pce')
        else:
            # Cross entropy on everything
            loss = nn.softmax_cross_entropy_with_logits(class_mask / class_sum, sims * 10)
        return loss
