import tensorflow as tf
from tensorflow.keras import layers

class GlobalBatchSims(layers.Layer):
    def call(self, inputs, **kwargs):
        feats1, feats2 = inputs
        replica_context = tf.distribute.get_replica_context()
        if replica_context is not None:
            global_feats2 = replica_context.all_gather(feats2, axis=0)
        else:
            strategy = tf.distribute.get_strategy()
            global_feats2 = strategy.gather(feats2, axis=0)
        feat_sims = tf.matmul(feats1, global_feats2, transpose_b=True)
        return feat_sims

class L2Normalize(layers.Layer):
    def call(self, inputs, **kwargs):
        tf.debugging.assert_rank(inputs, 2)
        return tf.nn.l2_normalize(inputs, axis=1)