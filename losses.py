import tensorflow as tf
from tensorflow.keras import losses


@tf.function
def supcon_loss(labels, feats1, feats2, partial):
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
