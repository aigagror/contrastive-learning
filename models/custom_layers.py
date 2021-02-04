import tensorflow as tf
from tensorflow.keras import layers
from typeguard import typechecked


class StandardizeImage(layers.Layer):

    def call(self, inputs, **kwargs):
        tf.debugging.assert_rank(inputs, 4)

        MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        inputs = tf.cast(inputs, self.dtype)
        inputs -= tf.constant(MEAN_RGB, shape=[1, 1, 1, 3], dtype=self.dtype)
        inputs /= tf.constant(STDDEV_RGB, shape=[1, 1, 1, 3], dtype=self.dtype)
        return inputs


class FeatViews(layers.Layer):
    def call(self, inputs, **kwargs):
        feats1, feats2 = inputs
        feats1 = tf.expand_dims(feats1, axis=1)
        feats2 = tf.expand_dims(feats2, axis=1)
        feat_views = tf.concat([feats1, feats2], axis=1)
        return feat_views


class GlobalBatchSims(layers.Layer):
    def call(self, inputs, **kwargs):
        feats1, feats2 = inputs

        if tf.distribute.in_cross_replica_context():
            strategy = tf.distribute.get_strategy()
            global_feats2 = strategy.gather(feats2, axis=0)
        else:
            replica_context = tf.distribute.get_replica_context()
            global_feats2 = replica_context.all_gather(feats2, axis=0)
        feat_sims = tf.matmul(feats1, global_feats2, transpose_b=True)
        return feat_sims


class L2Normalize(layers.Layer):
    def call(self, inputs, **kwargs):
        tf.debugging.assert_rank(inputs, 2)
        square_sum = tf.reduce_sum(tf.square(inputs), axis=1, keepdims=True)
        inv_norm = tf.math.rsqrt(tf.maximum(square_sum, 1e-12))
        return inputs * tf.stop_gradient(inv_norm)


class MeasureNorm(layers.Layer):
    def call(self, inputs, **kwargs):
        norms = tf.linalg.norm(inputs, axis=1)
        avg_norm = tf.reduce_mean(norms)
        self.add_metric(tf.cast(avg_norm, tf.float32), self.name)
        return inputs


class SpectralNormalization(tf.keras.layers.Wrapper):
    """Performs spectral normalization on weights.

    This wrapper controls the Lipschitz constant of the layer by
    constraining its spectral norm, which can stabilize the training of GANs.

    See [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957).

    Wrap `tf.keras.layers.Conv2D`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> conv2d = SpectralNormalization(tf.keras.layers.Conv2D(2, 2))
    >>> y = conv2d(x)
    >>> y.shape
    TensorShape([1, 9, 9, 2])

    Wrap `tf.keras.layers.Dense`:

    >>> x = np.random.rand(1, 10, 10, 1)
    >>> dense = SpectralNormalization(tf.keras.layers.Dense(10))
    >>> y = dense(x)
    >>> y.shape
    TensorShape([1, 10, 10, 10])

    Args:
      layer: A `tf.keras.layers.Layer` instance that
        has either `kernel` or `embeddings` attribute.
      power_iterations: `int`, the number of iterations during normalization.
    Raises:
      AssertionError: If not initialized with a `Layer` instance.
      ValueError: If initialized with negative `power_iterations`.
      AttributeError: If `layer` does not has `kernel` or `embeddings` attribute.
    """

    @typechecked
    def __init__(self, layer: tf.keras.layers, power_iterations: int = 1, **kwargs):
        super().__init__(layer, **kwargs)
        if power_iterations <= 0:
            raise ValueError(
                "`power_iterations` should be greater than zero, got "
                "`power_iterations={}`".format(power_iterations)
            )
        self.power_iterations = power_iterations
        self._initialized = False

    def build(self, input_shape):
        """Build `Layer`"""
        super().build(input_shape)
        input_shape = tf.TensorShape(input_shape)
        self.input_spec = tf.keras.layers.InputSpec(shape=[None] + input_shape[1:])

        if hasattr(self.layer, "kernel"):
            self.w = self.layer.kernel
        elif hasattr(self.layer, "embeddings"):
            self.w = self.layer.embeddings
        else:
            raise AttributeError(
                "{} object has no attribute 'kernel' nor "
                "'embeddings'".format(type(self.layer).__name__)
            )

        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name="sn_u",
            dtype=self.w.dtype,
        )

    def call(self, inputs, training=None):
        """Call `Layer`"""
        if training is None:
            training = tf.keras.backend.learning_phase()

        if training:
            self.normalize_weights()

        output = self.layer(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())

    @tf.function
    def normalize_weights(self):
        """Generate spectral normalized weights.

        This method will update the value of `self.w` with the
        spectral normalized value, so that the layer is ready for `call()`.
        """

        w = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u = self.u

        with tf.name_scope("spectral_normalize"):
            for _ in range(self.power_iterations):
                v = tf.math.l2_normalize(tf.matmul(u, w, transpose_b=True))
                u = tf.math.l2_normalize(tf.matmul(v, w))

            sigma = tf.matmul(tf.matmul(v, w), u, transpose_b=True)

            self.w.assign(self.w / sigma)
            self.u.assign(u)

    def get_config(self):
        config = {"power_iterations": self.power_iterations}
        base_config = super().get_config()
        return {**base_config, **config}


class Scale(layers.Layer):
    def __init__(self, initializer):
        super().__init__()
        self.scale = self.add_weight(shape=[], initializer=initializer, trainable=False, name="scale",
                                     dtype=self.dtype)

    def call(self, inputs, **kwargs):
        return self.scale * inputs


custom_objects = {
    'StandardizeImage': StandardizeImage,
    'FeatViews': FeatViews,
    'GlobalBatchSims': GlobalBatchSims,
    'L2Normalize': L2Normalize,
    'MeasureNorm': MeasureNorm,
    'Scale': Scale
}
