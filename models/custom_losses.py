import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import losses


class NoOp(losses.Loss):
    def call(self, y_true, y_pred):
        return 0


class ConLoss(losses.Loss):
    def process_y(self, y_true, y_pred):
        replica_context = tf.distribute.get_replica_context()

        # Feat views
        local_feat_views = tf.transpose(y_pred, [1, 0, 2])
        global_feat_views = replica_context.all_gather(local_feat_views, axis=1)

        # Predicted similarities
        feats1, all_feats2 = local_feat_views[0], global_feat_views[1]
        sims = tf.matmul(feats1, tf.stop_gradient(all_feats2), transpose_b=True)

        # Assert equal shapes
        tf.debugging.assert_shapes([
            (y_true, ['N', 'D']),
            (sims, ['N', 'D']),
        ])

        return y_true, sims

    def assert_inputs(self, y_true, y_pred):
        inst_mask = tf.cast((y_true == 2), tf.uint8)
        n_inst = tf.reduce_sum(inst_mask, axis=1)
        tf.debugging.assert_equal(n_inst, tf.ones_like(n_inst))

        tf.debugging.assert_shapes([
            (y_true, ['N', 'D']),
            (y_pred, ['N', 'D']),
        ])
        tf.debugging.assert_greater_equal(y_true, tf.zeros_like(y_true))
        tf.debugging.assert_less_equal(y_true, 2 * tf.ones_like(y_true))


class SimCLR(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)

        # Similarities
        sims = y_pred
        inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims * 10)
        return inst_loss


class SupCon(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        class_mask = tf.cast((y_true >= 1), dtype)
        labels, _ = tf.linalg.normalize(class_mask, ord=1, axis=1)

        # Similarities
        sims = y_pred
        supcon_loss = nn.softmax_cross_entropy_with_logits(labels, sims * 10)
        return supcon_loss


class PartialSupCon(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)
        partial_class_mask = tf.cast((y_true == 1), dtype)
        partial_class_sum = tf.math.reduce_sum(partial_class_mask, axis=1, keepdims=True)
        neg_mask = tf.cast((y_true == 0), dtype)

        # Similarities
        sims = y_pred * 10
        sims = sims - tf.stop_gradient(tf.reduce_max(sims, axis=1, keepdims=True))

        # Log probs
        exp = tf.math.exp(sims)
        neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
        partial_log_prob = sims - tf.math.log(neg_sum_exp + exp)
        tf.debugging.assert_less_equal(partial_log_prob, tf.zeros_like(partial_log_prob))

        # Partial class positive pairs log prob
        class_partial_log_prob = partial_class_mask * partial_log_prob
        class_partial_log_prob = tf.math.reduce_sum(class_partial_log_prob / (partial_class_sum + 1e-3), axis=1)
        partial_supcon_loss = -class_partial_log_prob

        loss = partial_supcon_loss + nn.softmax_cross_entropy_with_logits(inst_mask, sims)
        return loss


class BceSupCon(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)
        partial_class_mask = tf.cast((y_true == 1), dtype)
        neg_mask = tf.cast((y_true == 0), dtype)

        labels = inst_mask + (0.75 * partial_class_mask) + (0 * neg_mask)

        # Similarities
        sims = y_pred
        return losses.binary_crossentropy(labels, sims * 2 - 1, from_logits=False)


class MseSupCon(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)
        partial_class_mask = tf.cast((y_true == 1), dtype)
        neg_mask = tf.cast((y_true == 0), dtype)

        labels = inst_mask + (0.5 * partial_class_mask) + (-1 * neg_mask)

        # Similarities
        sims = y_pred
        return losses.mean_squared_error(labels, sims)
