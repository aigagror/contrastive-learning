import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import losses


class SimCLR(losses.Loss):

    def call(self, y_true, y_pred):
        dtype = y_pred.dtype
        bsz = len(y_true)

        # Masks
        inst_mask = tf.eye(bsz, dtype=dtype)

        # Similarities
        sims = y_pred

        inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims * 10)
        return inst_loss


class SupCon(losses.Loss):

    def call(self, y_true, y_pred):
        dtype = y_pred.dtype

        # Masks
        class_mask = tf.cast(y_true, dtype)
        class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

        # Similarities
        sims = y_pred
        supcon_loss = nn.softmax_cross_entropy_with_logits(class_mask / class_sum, sims * 10)
        return supcon_loss


class PartialSupCon(losses.Loss):

    def call(self, y_true, y_pred):
        dtype = y_pred.dtype
        bsz = len(y_true)

        # Masks
        inst_mask = tf.eye(bsz, dtype=dtype)
        class_mask = tf.cast(y_true, dtype)
        class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

        # Similarities
        sims = y_pred

        # Partial cross entropy on class similarities
        pos_mask = tf.maximum(inst_mask, class_mask)
        neg_mask = 1 - pos_mask

        exp = tf.math.exp(sims * 10)
        neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
        partial_log_prob = sims - tf.math.log(neg_sum_exp + exp)

        # Class positive pairs log prob (contains instance positive pairs too)
        class_partial_log_prob = class_mask * partial_log_prob
        class_partial_log_prob = tf.math.reduce_sum(class_partial_log_prob / class_sum, axis=1)
        partial_supcon_loss = -class_partial_log_prob

        return partial_supcon_loss
