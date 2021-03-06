import unittest

import main
import utils


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        # self.skipTest('time')
        pass

    def test_train_ce_load(self):
        args = '--data-id=mnist --backbone=affine ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 --train-steps=1 --val-steps=1 '
        args = utils.parser.parse_args(args.split())
        main.run(args)

        args = '--data-id=mnist --backbone=affine ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 --train-steps=1 --val-steps=1 ' \
               '--load'
        args = utils.parser.parse_args(args.split())
        main.run(args)

    def test_distributed_ce_from_load(self):
        args = '--data-id=mnist --backbone=affine ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 --train-steps=1 --val-steps=1  ' \
               '--multi-cpu'
        args = utils.parser.parse_args(args.split())
        main.run(args)

        args = '--data-id=mnist --backbone=affine ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 --train-steps=1 --val-steps=1 ' \
               '--multi-cpu --load'
        args = utils.parser.parse_args(args.split())
        main.run(args)

    def test_distributed_hiercon_from_load(self):
        args = '--data-id=mnist --backbone=affine --feat-norm=l2 ' \
               '--bsz=2 --lr=1e-3 --loss=hiercon ' \
               '--epochs=1 --train-steps=1 --val-steps=1  ' \
               '--multi-cpu'
        args = utils.parser.parse_args(args.split())
        main.run(args)

        args = '--data-id=mnist --backbone=affine --feat-norm=l2 ' \
               '--bsz=2 --lr=1e-3 --loss=hiercon ' \
               '--epochs=1 --train-steps=1 --val-steps=1 ' \
               '--multi-cpu --load'
        args = utils.parser.parse_args(args.split())
        main.run(args)


if __name__ == '__main__':
    unittest.main()
