import os
import tempfile
import unittest

import tensorflow as tf

import data
import models
import training
import utils


class TestTraining(unittest.TestCase):
    def test_compiles(self):
        for loss in ['ce', 'supcon', 'partial-supcon']:
            args = '--data=cifar10 --model=affine ' \
                   '--bsz=8 --lr=1e-3 ' \
                   f'--loss={loss} '
            args = utils.parser.parse_args(args.split())
            utils.setup(args)

            ds_train, _, _ = data.load_datasets(args)

            strategy = tf.distribute.MirroredStrategy(['CPU:0', 'CPU:1'])
            with strategy.scope():
                model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])
                training.compile_model(args, model)
            model.fit(ds_train, epochs=1, steps_per_epoch=1)

    def test_l2_reg(self):
        args = '--data=cifar10 --model=affine --weight-decay=1e-3 ' \
               '--bsz=8 --lr=1e-3 --loss=ce '
        args = utils.parser.parse_args(args.split())
        utils.setup(args)

        model = models.make_model(args, nclass=10, input_shape=[32, 32, 3])

        # Assert no regularization yet
        self.assertEqual(len(model.losses), 0)

        training.compile_model(args, model)

        # Assert regularization exists
        self.assertEqual(len(model.losses), 1)

        # Save
        tmp_model_path = os.path.join(tempfile.gettempdir(), 'model')
        model.save(tmp_model_path)

        loaded_model = tf.keras.models.load_model(tmp_model_path, models.all_custom_objects)

        # Assert regularization from fully loaded model
        self.assertEqual(len(loaded_model.losses), 1, loaded_model.losses)

        partial_load_model = tf.keras.models.load_model(tmp_model_path, compile=False)

        # Assert regularization from partially loaded model
        self.assertEqual(len(partial_load_model.losses), 1, partial_load_model.losses)


if __name__ == '__main__':
    unittest.main()
