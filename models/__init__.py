import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers, optimizers

from models import custom_layers, custom_losses
from models import small_resnet_v2


def make_model(args, nclass, input_shape):

    # Inputs
    input = keras.Input(input_shape, name='imgs')
    input2 = keras.Input(input_shape, name='imgs2')

    if args.cnn == 'resnet50v2':
        cnn = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape)
    elif args.cnn == 'small-resnet50v2':
        cnn = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape)
    else:
        raise Exception(f'unknown cnn model {args.cnn}')

    stand_img = custom_layers.StandardizeImage()

    # Feature maps
    feat_maps = cnn(stand_img(input))
    feat_maps2 = cnn(stand_img(input2))

    # Features
    if args.norm_feats:
        feats = custom_layers.L2Normalize(name='feats')(layers.GlobalAveragePooling2D()(feat_maps))
        feats2 = custom_layers.L2Normalize(name='feats2')(layers.GlobalAveragePooling2D()(feat_maps2))
    else:
        feats = layers.GlobalAveragePooling2D(name='feats')(feat_maps)
        feats2 = layers.GlobalAveragePooling2D(name='feats2')(feat_maps2)

    # Projected Features
    if args.norm_feats:
        proj_feats = custom_layers.L2Normalize(name='projection')(layers.Dense(128)(feats))
        proj_feats2 = custom_layers.L2Normalize(name='projection2')(layers.Dense(128)(feats2))
    else:
        proj_feats = layers.Dense(128, name='projection')(feats)
        proj_feats2 = layers.Dense(128, name='projection2')(feats2)

    # Batch similarities
    batch_sims = custom_layers.GlobalBatchSims(name='batch_sims')((proj_feats, proj_feats2))

    # Stop gradient at features?
    if args.method != 'ce':
        feats = tf.stop_gradient(feats)

    # Label logits
    prediction = layers.Dense(nclass, name='labels')(feats)

    # Model
    inputs = [input, input2]
    outputs = [prediction, batch_sims]

    model = keras.Model(inputs, outputs)

    # L2 regularization
    regularizer = keras.regularizers.l2(args.l2_reg)
    for module in model.submodules:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(module, attr):
                setattr(module, attr, regularizer)

    return model


def compile_model(args, model):
    # Optimizer
    opt = optimizers.SGD(args.lr, momentum=0.9)

    # Loss and metrics
    losses = {'labels': keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
    metrics = {'labels': 'acc'}
    if args.method == 'supcon':
        losses['batch_sims'] = custom_losses.SupCon()
        metrics['batch_sims'] = custom_losses.SupCon()
    elif args.method == 'partial-supcon':
        losses['batch_sims'] = [custom_losses.SimCLR(), custom_losses.PartialSupCon()]
        metrics['batch_sims'] = [custom_losses.SimCLR(), custom_losses.PartialSupCon()]
    elif args.method == 'bce-supcon':
        losses['batch_sims'] = custom_losses.BceSupCon()
        metrics['batch_sims'] = custom_losses.BceSupCon()
    elif args.method == 'simclr':
        losses['batch_sims'] = custom_losses.SimCLR()
    else:
        assert args.method == 'ce'

    # Compile
    model.compile(opt, losses, metrics, steps_per_execution=args.steps_exec)