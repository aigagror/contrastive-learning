import pickle

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, datasets


class ViewsTransform:
    def __init__(self, views):
        self.views = views

        # Sub-class must define the transform
        self.transform = None

    def __call__(self, x):
        ret = []
        for _ in range(self.views):
            ret.append(self.transform(x))
        return ret


class AugTransform(ViewsTransform):
    def __init__(self, views, imsize):
        super().__init__(views)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=imsize, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


class BaseTransform(ViewsTransform):
    def __init__(self, views):
        super().__init__(views)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


class ImgLabelTensorDataset(data.TensorDataset):
    def __init__(self, tensors, pil_transform, label_transform=None):
        super().__init__(*tensors)
        assert len(self.tensors) == 2
        self.targets = self.tensors[1].tolist()
        self.img_transform = transforms.Compose([transforms.ToPILImage(),
                                                 pil_transform])
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = self.img_transform(self.tensors[0][index])
        label = self.tensors[1][index]
        if self.label_transform is not None:
            label = self.label_transform(label)
        return (img, label)


def make_data_loader(dataset, batchsize, sampling, drop_last=False):
    if sampling == 'cb' or sampling == 'cr':
        targets = dataset.targets
        class_sample_count = np.unique(targets, return_counts=True)[1]

        if sampling == 'cb':
            weight = 1 / class_sample_count
        else:
            assert sampling == 'cr'
            weight = 1 / (class_sample_count ** 2)
        samples_weight = weight[targets]
        samples_weight = torch.from_numpy(samples_weight)
        sampler = data.WeightedRandomSampler(samples_weight, len(samples_weight))
        data_loader = data.DataLoader(dataset, batch_size=batchsize,
                                      sampler=sampler, num_workers=4,
                                      pin_memory=True, drop_last=drop_last)
    else:
        assert sampling == 'ib'
        data_loader = data.DataLoader(dataset, shuffle=True,
                                      batch_size=batchsize, num_workers=4,
                                      pin_memory=True, drop_last=drop_last)

    return data_loader


def mi_to_dataset(mi_data, pil_transform, label_transform=None):
    imgs, classes = [], []
    for c, class_imgs in enumerate(mi_data):
        class_imgs = class_imgs.transpose(0, 3, 1, 2) / 255
        for img in class_imgs:
            imgs.append(img)
            classes.append(c)
    imgs = np.array(imgs, dtype=np.float32)
    classes = np.array(classes, dtype=np.long)
    imgs = torch.tensor(imgs)
    classes = torch.tensor(classes)
    return ImgLabelTensorDataset((imgs, classes), pil_transform, label_transform)


def load_mini_imagenet_lt(args, train_transform, test_transform):
    mi_train = pickle.load(open(args.mi_lt_train, 'rb'))
    mi_test = pickle.load(open(args.mi_lt_test, 'rb'))

    train_dataset = mi_to_dataset(mi_train, train_transform)
    test_dataset = mi_to_dataset(mi_test, test_transform)
    return train_dataset, test_dataset


def load_data(args):
    if args.contrast:
        train_views = 2
        assert not args.fix_feats
    else:
        train_views = 1

    imsize_dict = {
        'mi-lt': 84,
        'cifar10': 32,
        'cifar100': 32
    }
    imsize = imsize_dict[args.data]
    train_transform = AugTransform(train_views, imsize)
    test_transform = BaseTransform(views=1)
    if args.data == 'mi-lt':
        train_dataset, test_dataset = load_mini_imagenet_lt(args, train_transform,
                                                            test_transform)
    elif args.data == 'cifar10':
        train_dataset = datasets.CIFAR10('./', train=True,
                                         transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10('./', train=False,
                                        transform=test_transform, download=True)
    elif args.data == 'cifar100':
        train_dataset = datasets.CIFAR100('./', train=True,
                                          transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100('./', train=False,
                                         transform=test_transform, download=True)
    else:
        raise Exception(f'Unknown data {args.data}')

    train_loader = make_data_loader(train_dataset, args.batchsize,
                                    args.sampling, drop_last=True)
    test_loader = make_data_loader(test_dataset, args.batchsize,
                                   sampling='ib')

    return train_loader, test_loader
