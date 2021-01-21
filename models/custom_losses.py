import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import losses


class SimCLR(losses.Loss):

    def call(self, y_true, y_pred):
        tf.debugging.assert_greater_equal(y_true, tf.zeros_like(y_true))
        tf.debugging.assert_less_equal(y_true, tf.ones_like(y_true))

        dtype = y_pred.dtype
        bsz, ncol = tf.shape(y_true)[:2]

        # Masks
        inst_mask = tf.eye(bsz, ncol, dtype=dtype)

        # Similarities
        sims = y_pred
        inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims * 10)
        return inst_loss


class SupCon(losses.Loss):

    def call(self, y_true, y_pred):
        tf.debugging.assert_greater_equal(y_true, tf.zeros_like(y_true))
        tf.debugging.assert_less_equal(y_true, tf.ones_like(y_true))

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
        tf.debugging.assert_greater_equal(y_true, tf.zeros_like(y_true))
        tf.debugging.assert_less_equal(y_true, tf.ones_like(y_true))

        dtype = y_pred.dtype
        bsz, ncol = tf.shape(y_true)[:2]

        # Masks
        inst_mask = tf.eye(bsz, ncol, dtype=dtype)
        class_mask = tf.cast(y_true, dtype)

        diag_part = tf.linalg.diag_part(class_mask)
        non_inst_class_mask = tf.linalg.set_diag(class_mask, tf.zeros_like(diag_part))
        non_inst_class_sum = tf.math.reduce_sum(non_inst_class_mask, axis=1, keepdims=True)

        # Similarities
        sims = y_pred * 10

        # Partial cross entropy on class similarities
        pos_mask = tf.maximum(inst_mask, class_mask)
        neg_mask = 1 - pos_mask

        exp = tf.math.exp(sims)
        neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
        partial_log_prob = sims - tf.math.log(neg_sum_exp + exp)

        # Class positive pairs log prob (contains instance positive pairs too)
        class_partial_log_prob = non_inst_class_mask * partial_log_prob
        class_partial_log_prob = tf.math.reduce_sum(class_partial_log_prob / (non_inst_class_sum + 1e-3), axis=1)
        partial_supcon_loss = -class_partial_log_prob

        return partial_supcon_loss
