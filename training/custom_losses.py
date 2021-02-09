import tensorflow as tf
from tensorflow import nn
from tensorflow.keras import losses


class ConLoss(losses.Loss):
    def __init__(self, temp, **kwargs):
        super().__init__(**kwargs)
        self.temp = temp

    def get_config(self):
        return {"temp": self.temp}

    def process_y(self, y_true, y_pred):
        tf.debugging.assert_shapes([(y_true, (None, 1))])
        replica_context = tf.distribute.get_replica_context()
        replica_id = replica_context.replica_id_in_sync_group

        # Feat views
        all_labels = replica_context.all_gather(y_true, axis=0)
        local_bsz, global_bsz = tf.shape(y_true)[0], tf.shape(all_labels)[0]
        batch_sims = tf.cast(all_labels == tf.transpose(all_labels), tf.uint8)
        batch_sims += tf.eye(global_bsz, dtype=tf.uint8)
        batch_sims = batch_sims[replica_id * local_bsz: (replica_id + 1) * local_bsz]

        all_y_pred = replica_context.all_gather(y_pred, axis=0)
        local_feat_views = tf.transpose(y_pred, [1, 0, 2])
        global_feat_views = tf.transpose(all_y_pred, [1, 0, 2])

        # Predicted similarities
        feats1, all_feats2 = local_feat_views[0], global_feat_views[1]
        sims = tf.matmul(feats1, all_feats2, transpose_b=True)

        # Assert equal shapes
        tf.debugging.assert_shapes([
            (batch_sims, ['N', 'D']),
            (sims, ['N', 'D']),
        ])

        return batch_sims, sims

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


class NoOp(losses.Loss):
    def call(self, y_true, y_pred):
        self.process_y(y_true, y_pred)
        return tf.constant(0, dtype=y_pred.dtype)


class SimCLR(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)

        # Similarities
        sims = y_pred
        inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims / self.temp)
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
        sims = y_pred / self.temp

        return nn.softmax_cross_entropy_with_logits(labels, sims)


class PartialSupCon(ConLoss):

    def call(self, y_true, y_pred):
        y_true, y_pred = self.process_y(y_true, y_pred)
        self.assert_inputs(y_true, y_pred)
        dtype = y_pred.dtype

        # Masks
        inst_mask = tf.cast((y_true == 2), dtype)
        partial_class_mask = tf.cast((y_true == 1), dtype)
        partial_class_sum = tf.math.reduce_sum(partial_class_mask, axis=1)
        partial_mask = tf.cast((y_true <= 1), dtype)

        # Similarities
        sims = y_pred / self.temp
        sims = sims - tf.stop_gradient(tf.reduce_max(sims, axis=1, keepdims=True))

        # Log probs
        exp = tf.math.exp(sims)
        partial_sum_exp = tf.math.reduce_sum(exp * partial_mask, axis=1, keepdims=True)
        partial_log_prob = sims - tf.math.log(partial_sum_exp + 1e-5)

        # Partial class positive pairs log prob
        class_partial_log_prob = partial_class_mask * partial_log_prob
        tf.debugging.assert_less_equal(class_partial_log_prob, tf.zeros_like(class_partial_log_prob))
        class_partial_log_prob = tf.math.reduce_sum(class_partial_log_prob, axis=1)
        class_partial_log_prob = tf.math.divide_no_nan(class_partial_log_prob, partial_class_sum)
        partial_supcon_loss = -class_partial_log_prob

        inst_loss = nn.softmax_cross_entropy_with_logits(inst_mask, sims)
        loss = partial_supcon_loss + inst_loss
        return loss


custom_objects = {
    'NoOp': NoOp,
    'SimCLR': SimCLR,
    'SupCon': SupCon,
    'PartialSupCon': PartialSupCon,
}
