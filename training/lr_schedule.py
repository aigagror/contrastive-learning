import tensorflow as tf


class PiecewiseConstantDecayWithWarmup(tf.keras.optimizers.schedules.PiecewiseConstantDecay):
    """Piecewise constant decay with warmup schedule."""

    def __init__(self, lr, steps_per_epoch, warmup_epochs, epoch_boundaries, name=None):
        self.lr = lr
        self.step_boundaries = [steps_per_epoch * x for x in epoch_boundaries]
        self.lr_values = [self.lr] + [self.lr * (10 ** (-i)) for i in range(1, len(epoch_boundaries) + 1)]
        self.warmup_steps = warmup_epochs * steps_per_epoch

        super().__init__(self.step_boundaries, self.lr_values, name)

    def __call__(self, step):
        if step < self.warmup_steps:
            return self.lr * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
        else:
            return super().__call__(step)

    def get_config(self):
        parent_config = super().get_config()
        subclass_config = {
            'lr': self.lr,
            'step_boundaries': self.step_boundaries,
            'lr_values': self.lr_values,
            'warmup_steps': self.warmup_steps
        }

        return {**parent_config, **subclass_config}
