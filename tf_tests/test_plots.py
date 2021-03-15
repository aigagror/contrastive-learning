import logging
import unittest

import tensorflow_datasets as tfds

import data
import models
import plots
import utils


class TestPlots(unittest.TestCase):
    def basic_usage(self):
        args = '--data-id=mnist --backbone=affine ' \
               '--bsz=8 --lr=1e-3 ' \
               '--loss=hiercon --multi-cpu'

        args = utils.parser.parse_args(args.split())
        strategy = utils.setup(args)

        _, ds_info = tfds.load(args.data_id, try_gcs=True, data_dir='gs://aigagror/datasets', with_info=True)
        train_augment_config, val_augment_config = utils.load_augment_configs(args)
        ds_train, ds_val = data.load_distributed_datasets(args, ds_info, strategy, train_augment_config,
                                                          val_augment_config)

        with strategy.scope():
            model = models.make_model(args, ds_info.features['label'].num_classes, ds_info.features['image'].shape)

        return args, strategy, model, ds_train

    def test_hist(self):
        args, strategy, model, ds_val = self.basic_usage()
        plots.plot_hist_sims(args, strategy, model, ds_val)
        logging.info(f'hist saved to {args.out}')

    def test_distributed_hist(self):
        args, strategy, model, ds_val = self.basic_usage()
        plots.plot_hist_sims(args, strategy, model, ds_val)
        logging.info(f'hist saved to {args.out}')

    def test_tsne(self):
        args, strategy, model, ds_val = self.basic_usage()
        plots.plot_tsne(args, strategy, model, ds_val)
        logging.info(f'tsne saved to {args.out}')


if __name__ == '__main__':
    unittest.main()
