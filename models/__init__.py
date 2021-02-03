import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers

from models import custom_layers, custom_losses
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


def make_model(args, nclass, input_shape):
    # Weight decay
    if args.weight_decay is not None and args.weight_decay > 0:
        regularizer = keras.regularizers.L2(args.weight_decay / 2)
    else:
        regularizer = None
    if args.optimizer == 'lamb':
        regularizer = None
        print('Adding weight decay via the LAMB optimizer instead of Keras regularization')

    # Inputs
    input = keras.Input(input_shape, name='imgs')
    input2 = keras.Input(input_shape, name='imgs2')

    if args.backbone == 'resnet50v2':
        backbone = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
        if args.data == 'cifar':
            print('WARNING: Using standard resnet on small dataset')
    elif args.backbone == 'small-resnet50v2':
        backbone = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')
        if args.data == 'imagenet':
            print('WARNING: Using small resnet on large dataset')
    elif args.backbone == 'resnet50':
        backbone = applications.ResNet50(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
    elif args.backbone == 'affine':
        backbone = keras.Sequential([
            layers.Conv2D(128, 3, kernel_regularizer=regularizer, bias_regularizer=regularizer),
            layers.GlobalAveragePooling2D()
        ])
    else:
        raise Exception(f'unknown model {args.backbone}')

    # Add weight decay to backbone
    backbone = add_regularization_with_reset(backbone, regularizer)

    # Standardize input
    stand_img = custom_layers.StandardizeImage()

    # Features
    raw_feats = backbone(stand_img(input))
    raw_feats2 = backbone(stand_img(input2))

    # Normalize?
    if args.feat_norm == 'l2':
        # L2 normalize
        feats = custom_layers.L2Normalize()(raw_feats)
        feats2 = custom_layers.L2Normalize()(raw_feats2)
    elif args.feat_norm == 'sn':
        # Average L2 norm with BN
        feats = layers.BatchNormalization(scale=False, center=False)(raw_feats)
        feats2 = layers.BatchNormalization(scale=False, center=False)(raw_feats2)
        feats_scale = tf.math.rsqrt(tf.cast(feats.shape[-1], tf.float32))
        feats, feats2 = feats * feats_scale, feats2 * feats_scale
    else:
        # No normalization
        feats = raw_feats
        feats2 = raw_feats2

    # Measure the norms of the features
    feats = custom_layers.MeasureNorm(name='feat_norm')(feats)

    # Name the features
    feats = layers.Activation('linear', name='feats')(feats)
    feats2 = layers.Activation('linear', name='feats2')(feats2)

    # Projected features
    projection = layers.Dense(128, name='projection', kernel_regularizer=regularizer, bias_regularizer=regularizer)

    # Normalize?
    if args.feat_norm == 'l2':
        # L2 normalize
        proj_feats = projection(feats)
        proj_feats2 = projection(feats2)
        proj_feats = custom_layers.L2Normalize()(proj_feats)
        proj_feats2 = custom_layers.L2Normalize()(proj_feats2)
    elif args.feat_norm == 'sn':
        # Spectral normalize
        projection = custom_layers.SpectralNormalization(projection)
        proj_feats = projection(feats)
        proj_feats2 = projection(feats2)
    else:
        # No normalization
        proj_feats = projection(feats)
        proj_feats2 = projection(feats2)

    # Measure the norms of the projected features
    proj_feats = custom_layers.MeasureNorm(name='proj_norm')(proj_feats)

    # Name the projected features
    proj_feats = layers.Activation('linear', name='proj_feats')(proj_feats)
    proj_feats2 = layers.Activation('linear', name='proj_feats2')(proj_feats2)

    # Feature views
    proj_views = custom_layers.FeatViews(name='contrast', dtype=tf.float32)((proj_feats, proj_feats2))

    # Label logits
    prediction = layers.Dense(nclass, name='labels', kernel_regularizer=regularizer, bias_regularizer=regularizer,
                              dtype=tf.float32)(feats)

    # Model
    inputs = [input]
    outputs = [prediction]
    if args.loss != 'ce':
        inputs.append(input2)
        outputs.append(proj_views)

    model = keras.Model(inputs, outputs)

    return model


all_custom_objects = {**custom_losses.custom_objects, **custom_layers.custom_objects}
