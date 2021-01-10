import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import applications, layers, losses


def compute_supcon_loss(labels, feats1, feats2, partial):
    bsz = len(labels)
    labels = tf.expand_dims(labels, 1)
    dtype = feats1.dtype

    # Masks
    inst_mask = tf.eye(bsz, dtype=dtype)
    class_mask = tf.cast(labels == tf.transpose(labels), dtype)
    class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

    # Similarities
    sims = tf.matmul(feats1, tf.transpose(feats2))

    if partial:
        # Cross entropy on instance similarities
        inst_loss = losses.categorical_crossentropy(inst_mask, sims * 10, from_logits=True)

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
    else:
        # Cross entropy on everything
        loss = losses.categorical_crossentropy(class_mask / class_sum, sims * 10, from_logits=True)

    return loss


class ContrastModel(keras.Model):
    def __init__(self, args, nclass):
        super().__init__()
        self.args = args

        self.preprocess = applications.resnet_v2.preprocess_input
        self.cnn = applications.ResNet50V2(weights=None, include_top=False)

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
            self.load_weights(args.model_out)
        else:
            print(f'starting with new model weights')

    def norm_feats(self, img):
        x = self.preprocess(img)
        x = self.cnn(x)
        x = self.avg_pool(x)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def norm_project(self, feats):
        x = self.projection(feats)
        x, _ = tf.linalg.normalize(x, axis=-1)
        return x

    def call(self, input, **kwargs):
        if self.args.method == 'ce':
            feats = self.norm_feats(input['imgs'])
            pred_logits = self.classifier(feats)
        else:
            assert self.args.method.startswith('supcon')
            partial = self.args.method.endswith('-pce')

            feats = self.norm_feats(input['imgs'])
            proj_feats = self.norm_project(feats)

            feats2 = self.norm_feats(input['imgs2'])
            proj_feats2 = self.norm_project(feats2)

            supcon_loss = compute_supcon_loss(input['labels'], proj_feats, proj_feats2, partial)
            supcon_loss = nn.compute_average_loss(supcon_loss, global_batch_size=self.args.bsz)
            self.add_loss(supcon_loss)

            pred_logits = self.classifier(tf.stop_gradient(feats))

        ce_loss = losses.sparse_categorical_crossentropy(input['labels'], pred_logits, from_logits=True)
        ce_loss = nn.compute_average_loss(ce_loss, global_batch_size=self.args.bsz)
        self.add_loss(ce_loss)
        self.add_metric(ce_loss, 'cross-entropy')

        pred = tf.argmax(pred_logits, axis=1)
        return pred
