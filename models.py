import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck


## Models

def count_num_params(model):
    return sum(p.numel() for p in model.parameters())


#### Encoder

class EncoderResnet(ResNet):
    def __init__(self, arch):
        if arch == 'resnet18':
            super().__init__(BasicBlock, [2, 2, 2, 2])
        else:
            assert arch == 'resnet50'
            super().__init__(Bottleneck, [3, 4, 6, 3])
        del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.normalize(x)
        return x


#### Projection

class Projection(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(),
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        x = self.main(x)
        x = F.normalize(x)
        return x


#### Contrast model

class ContrastModel(nn.Module):
    def __init__(self, arch, nclass, wn):
        super().__init__()
        self.nclass, self.wn = nclass, wn
        self.encoder = EncoderResnet(arch)
        self.projection = Projection(512, 128)
        self.reset_classifier()

    def feats(self, img_views):
        feat_views = [self.encoder(imgs) for imgs in img_views]
        return feat_views

    def project(self, feat_views):
        project_views = [self.projection(feats) for feats in feat_views]
        return project_views

    def reset_classifier(self):
        if self.wn:
            self.fc = weight_norm(nn.Linear(512, self.nclass, bias=False))
        else:
            self.fc = nn.Linear(512, self.nclass)


#### Model utilities

def make_model(args):
    model = ContrastModel(args.model, nclass=100, wn=False)

    if args.fix_feats:
        model.encoder.requires_grad_(False)
        print(f"Fixed encoder features")

    # Model summary
    print(f'Model {model.__class__.__name__}: {count_num_params(model)} parameters')

    return model


def optional_load_wts(args, model, model_path):
    if os.path.exists(model_path) and args.load:
        model.load_state_dict(torch.load(model_path))
        print(f"loaded saved weights from {model_path}")
    else:
        print("new weights")
