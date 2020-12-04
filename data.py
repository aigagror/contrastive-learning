import pickle

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms, datasets


## Data

#### Transforms

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


#### Datasets and dataloaders

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


def load_mi_class_imgs():
    train_pkl = pickle.load(open("/gdrive/My Drive/datasets/miniimagenet/train.pkl", "rb"))
    val_pkl = pickle.load(open("/gdrive/My Drive/datasets/miniimagenet/val.pkl", "rb"))
    test_pkl = pickle.load(open("/gdrive/My Drive/datasets/miniimagenet/test.pkl", "rb"))

    imgs_train = train_pkl['image_data'].reshape(64, 600, 84, 84, 3)
    imgs_val = val_pkl['image_data'].reshape(16, 600, 84, 84, 3)
    imgs_test = test_pkl['image_data'].reshape(20, 600, 84, 84, 3)

    class_imgs = np.concatenate([imgs_train, imgs_val, imgs_test])
    class_imgs = class_imgs.astype(np.float32)
    class_imgs = class_imgs.transpose(0, 1, 4, 2, 3) / 255
    return class_imgs


def mi_to_dataset(mi_path, pil_transform, label_transform=None):
    all_class_imgs = load_mi_class_imgs()
    mi_data = pickle.load(open(mi_path, 'rb'))
    imgs, classes = [], []
    for c, class_idcs in enumerate(mi_data):
        imgs.append(all_class_imgs[c, class_idcs])
        classes.append(np.full(class_idcs.shape, c))

    imgs = np.concatenate(imgs)
    classes = np.concatenate(classes).astype(np.long)

    imgs = torch.tensor(imgs)
    classes = torch.tensor(classes)
    return ImgLabelTensorDataset((imgs, classes), pil_transform, label_transform)


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


#### Load data

def load_data(args):
    imsize_dict = {
        'mi-bal': 84,
        'mi-lt': 84,
        'cifar10': 32,
        'cifar100': 32
    }
    imsize = imsize_dict[args.data]
    train_transform = AugTransform(args.nviews, imsize)
    test_transform = BaseTransform(views=1)
    if args.data.startswith('mi'):
        # Mini Imagenet
        if args.data.endswith('-lt'):
            # Long tail version
            train_path = "/gdrive/My Drive/datasets/miniimagenet/custom-lt/train.pkl"
            test_path = "/gdrive/My Drive/datasets/miniimagenet/custom-lt/test.pkl"
        else:
            assert args.data.endswith('mi-bal')
            # Regular balanced version
            train_path = "/gdrive/My Drive/datasets/miniimagenet/custom-balanced/train.pkl"
            test_path = "/gdrive/My Drive/datasets/miniimagenet/custom-balanced/test.pkl"

        train_dataset = mi_to_dataset(train_path, train_transform)
        test_dataset = mi_to_dataset(test_path, test_transform)
    elif args.data == 'cifar10':
        # CIFAR 10
        train_dataset = datasets.CIFAR10('./', train=True,
                                         transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10('./', train=False,
                                        transform=test_transform, download=True)
    elif args.data == 'cifar100':
        # CIFAR 100
        train_dataset = datasets.CIFAR100('./', train=True,
                                          transform=train_transform, download=True)
        test_dataset = datasets.CIFAR100('./', train=False,
                                         transform=test_transform, download=True)
    else:
        raise Exception(f'Unknown data {args.data}')

    # Data loader
    train_loader = make_data_loader(train_dataset, args.batchsize,
                                    args.sampling, drop_last=True)
    test_loader = make_data_loader(test_dataset, args.batchsize,
                                   sampling='ib')

    return train_loader, test_loader
