import logging

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers

from models import custom_layers
from models import small_resnet_v2


def add_regularization_with_reset(model, regularizer):
    for module in model.submodules:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(module, attr):
                setattr(module, attr, regularizer)

    # When we change the layers attributes, the change only happens in the model config file
    model_json = model.to_json()

    # load the model from the config
    model = tf.keras.models.model_from_json(model_json)

    return model


def optional_normalize(norm, feats1, feats2):
    if norm == 'l2':
        # L2 normalize
        feats = custom_layers.L2Normalize()(feats1)
        feats2 = custom_layers.L2Normalize()(feats2)
    elif norm == 'bn':
        # Average L2 norm with BN
        batchnorm = layers.BatchNormalization(scale=False, center=False)
        feats, feats2 = batchnorm(feats1), batchnorm(feats2)

        # Scale down by sqrt of feature dimension
        feats_scale = 1 / (feats1.shape[-1] ** 0.5)
        scale = custom_layers.Scale(feats_scale)
        feats, feats2 = scale(feats), scale(feats2)
    else:
        # No normalization
        assert norm is None or norm == 'sn'
        feats = feats1
        feats2 = feats2
    return feats, feats2


def make_model(args, nclass, input_shape):
    # Weight decay
    if args.weight_decay is not None and args.weight_decay > 0:
        regularizer = keras.regularizers.L2(args.weight_decay / 2)
    else:
        regularizer = None
    if args.optimizer == 'lamb':
        regularizer = None
        logging.info('adding weight decay via the LAMB optimizer instead of Keras regularization')

    # Inputs
    input = keras.Input(input_shape, name='image')
    input2 = keras.Input(input_shape, name='image2')

    if args.backbone == 'resnet50v2':
        backbone = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
        if input_shape[0] < 224:
            logging.warning('using standard resnet on small dataset')
    elif args.backbone == 'small-resnet50v2':
        backbone = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')
        if input_shape[0] >= 224:
            logging.warning('using small resnet on large dataset')
    elif args.backbone == 'resnet50':
        backbone = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
    elif args.backbone == 'affine':
        backbone = keras.Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Dense(1)
        ])
    else:
        raise Exception(f'unknown model {args.backbone}')

    # Add weight decay to backbone
    backbone = add_regularization_with_reset(backbone, regularizer)

    # Standardize input
    stand_img = custom_layers.StandardizeImage()

    # Features
    raw_feats, raw_feats2 = backbone(stand_img(input)), backbone(stand_img(input2))

    # Normalize features?
    feats, feats2 = optional_normalize(args.feat_norm, raw_feats, raw_feats2)

    # Name the features
    feats = custom_layers.Identity(name='feats')(feats)
    feats2 = custom_layers.Identity(name='feats2')(feats2)

    # Measure the norms of the features
    feats = custom_layers.MeasureNorm(name='feat_norm')(feats)

    # Projection
    if args.proj_dim is None or args.proj_dim <= 0:
        projection = custom_layers.Identity(name='projection')
    else:
        projection = layers.Dense(args.proj_dim, name='projection', use_bias=False,
                                  kernel_regularizer=regularizer if args.proj_norm is None else None,
                                  bias_regularizer=regularizer if args.proj_norm is None else None)
    if args.proj_norm == 'sn':
        projection = custom_layers.SpectralNormalization(projection, name='sn_projection')

    # Projected features
    proj_feats, proj_feats2 = projection(feats), projection(feats2)

    # Normalize projected features?
    proj_feats, proj_feats2 = optional_normalize(args.proj_norm, proj_feats, proj_feats2)

    # Name the projected features
    proj_feats = custom_layers.Identity(name='proj_feats')(proj_feats)
    proj_feats2 = custom_layers.Identity(name='proj_feats2')(proj_feats2)

    # Measure the norms of the projected features
    proj_feats = custom_layers.MeasureNorm(name='proj_norm')(proj_feats)

    # Feature views
    proj_views = custom_layers.FeatViews(name='contrast', dtype=tf.float32)((proj_feats, proj_feats2))

    # Label logits
    prediction = layers.Dense(nclass, name='label', kernel_regularizer=regularizer, bias_regularizer=regularizer,
                              dtype=tf.float32)(feats)

    # Model
    inputs = [input]
    outputs = [prediction]
    if args.loss != 'ce':
        inputs.append(input2)
        outputs.append(proj_views)

    model = keras.Model(inputs, outputs)

    return model
