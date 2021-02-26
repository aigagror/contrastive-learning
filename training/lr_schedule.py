import tensorflow as tf


class PiecewiseConstantDecayWithWarmup(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    """Piecewise constant decay with warmup schedule."""

    def __init__(self, lr, steps_per_epoch, warmup_epochs, epoch_boundaries, start_step=0):
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.warmup_epochs = warmup_epochs
        self.epoch_boundaries = epoch_boundaries
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.start_step = start_step

        if epoch_boundaries is not None:
            boundaries = [steps_per_epoch * x for x in epoch_boundaries]
        else:
            boundaries = [float('inf')]
        values = [self.lr] + [self.lr * (10 ** (-i)) for i in range(1, len(boundaries) + 1)]
        super().__init__(boundaries, values)

    def __call__(self, step):
        step = self.start_step + step
        lr = tf.cond(step < self.warmup_steps,
                     lambda: self.lr * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)),
                     lambda: super(PiecewiseConstantDecayWithWarmup, self).__call__(step))
        return lr

    def get_config(self):
        return {
            'lr': self.lr,
            'steps_per_epoch': self.steps_per_epoch,
            'warmup_epochs': self.warmup_epochs,
            'epoch_boundaries': self.epoch_boundaries,
            'start_step': self.start_step,
        }


class WarmUpAndCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies a warmup schedule on a given learning rate decay schedule."""

    def __init__(self, lr, steps_epoch, warmup_epochs, epochs, name=None):
        super(WarmUpAndCosineDecay, self).__init__()
        self.lr = lr
        self.steps_epoch = steps_epoch
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self._name = name

    def __call__(self, step):
        with tf.name_scope(self._name or 'WarmUpAndCosineDecay'):
            warmup_steps = int(
                round(self.warmup_epochs * self.steps_epoch))
            learning_rate = tf.cast(step, tf.float32) / warmup_steps * self.lr if warmup_steps else self.lr

            # Cosine decay learning rate schedule
            total_steps = self.steps_epoch * self.epochs
            cosine_decay = tf.keras.experimental.CosineDecay(
                self.lr, total_steps - warmup_steps)
            learning_rate = tf.where(step < warmup_steps, learning_rate,
                                     cosine_decay(step - warmup_steps))

            return learning_rate

    def get_config(self):
        return {
            'lr': self.lr,
            'steps_epoch': self.steps_epoch,
            'warmup_epochs': self.warmup_epochs,
            'epochs': self.epochs,
        }


custom_objects = {
    'PiecewiseConstantDecayWithWarmup': PiecewiseConstantDecayWithWarmup,
    'WarmUpAndCosineDecay': WarmUpAndCosineDecay,
}
