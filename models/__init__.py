import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, optimizers

from models import custom_layers, custom_losses
from models import small_resnet_v2


def make_model(args, nclass, input_shape):

    # Inputs
    input = keras.Input(input_shape, name='imgs')
    input2 = keras.Input(input_shape, name='imgs2')

    if args.model.startswith('resnet50v2'):
        resnet = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
        if args.data.startswith('cifar'):
            print('WARNING: Using standard resnet on small dataset')
    elif args.model.startswith('small-resnet50v2'):
        resnet = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape)
        if args.data == 'imagenet':
            print('WARNING: Using small resnet on large dataset')
    else:
        raise Exception(f'unknown model {args.model}')

    stand_img = custom_layers.StandardizeImage()

    # Feature maps
    feat_maps = resnet(stand_img(input))
    feat_maps2 = resnet(stand_img(input2))

    # Features
    if args.model.endswith('-norm'):
        feats = custom_layers.L2Normalize(name='feats')(layers.GlobalAveragePooling2D()(feat_maps))
        feats2 = custom_layers.L2Normalize(name='feats2')(layers.GlobalAveragePooling2D()(feat_maps2))
    else:
        feats = layers.GlobalAveragePooling2D(name='feats')(feat_maps)
        feats2 = layers.GlobalAveragePooling2D(name='feats2')(feat_maps2)

    # Projected Features
    if args.model.endswith('-norm'):
        proj_feats = custom_layers.L2Normalize(name='projection')(layers.Dense(128)(feats))
        proj_feats2 = custom_layers.L2Normalize(name='projection2')(layers.Dense(128)(feats2))
    else:
        proj_feats = layers.Dense(128, name='projection')(feats)
        proj_feats2 = layers.Dense(128, name='projection2')(feats2)

    # Feature views
    proj_views = custom_layers.FeatViews(name='contrast')((proj_feats, proj_feats2))

    # Stop gradient at features?
    if args.method != 'ce':
        feats = tf.stop_gradient(feats)

    # Label logits
    prediction = layers.Dense(nclass, name='labels')(feats)

    # Model
    inputs = [input, input2]
    outputs = [prediction, proj_views]

    model = keras.Model(inputs, outputs)

    # L2 regularization
    if args.l2_reg is not None:
        print('added l2 regularization')
        regularizer = keras.regularizers.l2(args.l2_reg)
        for module in model.submodules:
            for attr in ['kernel_regularizer', 'bias_regularizer']:
                if hasattr(module, attr):
                    setattr(module, attr, regularizer)
    else:
        print('no l2 regularization')

    return model


def compile_model(args, model):
    # Optimizer
    opt = optimizers.SGD(args.lr, momentum=0.9)

    # Loss and metrics
    losses = {'labels': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
    metrics = {'labels': 'acc'}

    contrast_loss_dict = {
        'supcon': custom_losses.SupCon(),
        'partial-supcon': custom_losses.PartialSupCon(),
        'bce-supcon': custom_losses.BceSupCon(),
        'mse-supcon': custom_losses.MseSupCon(),
        'simclr': custom_losses.SimCLR()
    }
    if args.method in contrast_loss_dict:
        losses['contrast'] = contrast_loss_dict[args.method]

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)