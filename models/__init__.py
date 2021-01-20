import os

import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import applications, layers, metrics, losses
from models import small_resnet_v2

class ContrastModel(keras.Model):
    def __init__(self, args, nclass, input_shape):
        super().__init__()
        self.args = args

        if args.cnn == 'resnet50v2':
            self._cnn = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
        elif args.cnn == 'small-resnet50v2':
            self._cnn = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape)
        else:
            raise Exception(f'unknown cnn model {args.cnn}')

        self._avg_pool = layers.GlobalAveragePooling2D()
        self._projection = layers.Dense(128, name='projection')
        self._classifier = layers.Dense(nclass, name='classifier')

        # L2 regularization
        regularizer = keras.regularizers.l2(args.l2_reg)
        for module in self.submodules:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(module, attr):
                    setattr(module, attr, regularizer)

        # Load weights?
        if args.load:
            # Call to build weights, then load
            print(f'loaded previously saved model weights')
            self.load_weights(os.path.join(args.out, 'model'))
        else:
            print(f'starting with new model weights')

    def features(self, img):
        x = tf.cast(img, self.args.dtype) / 127.5 - 1
        x = self._cnn(x)
        x = self._avg_pool(x)
        if self.args.norm_feats:
            x = tf.linalg.l2_normalize(x, axis=-1)
        return x

    def projection(self, feats):
        x = self._projection(feats)
        if self.args.norm_feats:
            x = tf.linalg.l2_normalize(x, axis=-1)
        return x

    def call(self, input, **kwargs):
        if self.args.method == 'ce':
            feats = self.features(input['imgs'])
            pred_logits = self._classifier(feats)
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
            self.add_metric(tf.cast(supcon_loss, tf.float32), 'supcon')

            pred_logits = self._classifier(tf.stop_gradient(feats))

        # Cross entropy and accuracy
        pred_logits = tf.cast(pred_logits, tf.float32)
        ce_loss = losses.sparse_categorical_crossentropy(input['labels'], pred_logits, from_logits=True)
        ce_loss = tf.reduce_mean(ce_loss)
        acc = metrics.sparse_categorical_accuracy(input['labels'], pred_logits)
        acc = tf.reduce_mean(acc)
        self.add_loss(ce_loss)
        self.add_metric(tf.cast(ce_loss, tf.float32), 'ce')
        self.add_metric(tf.cast(acc, tf.float32), 'acc')

        # Prediction
        pred = tf.argmax(pred_logits, axis=1)
        return pred

    def compute_supcon_loss(self, labels, feats1, feats2, partial):
        tf.debugging.assert_shapes([(labels, [None, 1])])
        dtype = feats1.dtype

        # Gather everything
        replica_context = tf.distribute.get_replica_context()
        labels = replica_context.all_gather(labels, axis=0)
        feats1 = replica_context.all_gather(feats1, axis=0)
        feats2 = replica_context.all_gather(feats2, axis=0)
        bsz = len(labels)

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
            self.add_metric(tf.cast(tf.reduce_mean(inst_loss), tf.float32), 'inst-ce')
            self.add_metric(tf.cast(tf.reduce_mean(class_loss), tf.float32), 'class-pce')
        else:
            # Cross entropy on everything
            loss = nn.softmax_cross_entropy_with_logits(class_mask / class_sum, sims * 10)
        return loss
