import tensorflow as tf
from tensorflow.keras import losses


@tf.function
def supcon_loss(labels, feats1, feats2, partial):
    bsz = len(labels)
    labels = tf.expand_dims(labels, 1)

    # Masks
    inst_mask = tf.eye(bsz, dtype=tf.float16)
    class_mask = tf.cast(labels == tf.transpose(labels), tf.float16)
    class_sum = tf.math.reduce_sum(class_mask, axis=1, keepdims=True)

    # Similarities
    sims = tf.matmul(feats1, tf.transpose(feats2))

    if partial:
        # Partial cross entropy
        pos_mask = tf.maximum(inst_mask, class_mask)
        neg_mask = 1 - pos_mask

        exp = tf.math.exp(sims * 10)
        neg_sum_exp = tf.math.reduce_sum(exp * neg_mask, axis=1, keepdims=True)
        log_prob = sims - tf.math.log(neg_sum_exp + exp)

        # Class positive pairs log prob (contains instance positive pairs too)
        class_log_prob = class_mask * log_prob
        class_log_prob = tf.math.reduce_sum(class_log_prob / class_sum, axis=1)

        loss = -class_log_prob
    else:
        # Cross entropy
        loss = losses.categorical_crossentropy(class_mask / class_sum, sims * 10,
                                               from_logits=True)
    return loss
