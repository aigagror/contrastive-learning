import unittest

import main
import utils


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.skipTest('takes too long')
        pass

    def test_train_ce_load(self):
        args = '--data=fake-cifar10 --model=affine ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 --debug '
        args = utils.parser.parse_args(args.split())
        main.run(args)

        args = '--data=fake-cifar10 --load ' \
               '--bsz=2 --lr=1e-3 --loss=ce ' \
               '--epochs=1 '
        args = utils.parser.parse_args(args.split())
        main.run(args)

    def test_distributed_partial_supcon_from_load(self):
        args = '--data=fake-cifar10 --model=affine-norm ' \
               '--bsz=2 --lr=1e-3 --loss=partial-supcon ' \
               '--epochs=1 --multi-cpu --debug '
        args = utils.parser.parse_args(args.split())
        main.run(args)

        args = '--data=fake-cifar10 --load ' \
               '--bsz=2 --lr=1e-3 --loss=partial-supcon ' \
               '--epochs=1 --multi-cpu '
        args = utils.parser.parse_args(args.split())
        main.run(args)


if __name__ == '__main__':
    unittest.main()
