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
        regularizer = keras.regularizers.L2(args.weight_decay)
    else:
        regularizer = None

    # Inputs
    input = keras.Input(input_shape, name='imgs')
    input2 = keras.Input(input_shape, name='imgs2')

    if args.model.startswith('resnet50v2'):
        backbone = applications.ResNet50V2(weights=None, include_top=False, input_shape=input_shape, pooling='avg')
        if args.data.startswith('cifar'):
            print('WARNING: Using standard resnet on small dataset')
    elif args.model.startswith('small-resnet50v2'):
        backbone = small_resnet_v2.SmallResNet50V2(include_top=False, input_shape=input_shape, pooling='avg')
        if args.data == 'imagenet':
            print('WARNING: Using small resnet on large dataset')
    elif args.model == 'affine':
        backbone = keras.Sequential([
            layers.Conv2D(128, 3, kernel_regularizer=regularizer, bias_regularizer=regularizer),
            layers.GlobalAveragePooling2D()
        ])
    else:
        raise Exception(f'unknown model {args.model}')

    # Add weight decay to backbone
    backbone = add_regularization_with_reset(backbone, regularizer)

    # Standardize input
    stand_img = custom_layers.StandardizeImage()

    # Feature maps
    feat_maps = backbone(stand_img(input))
    feat_maps2 = backbone(stand_img(input2))

    # Features
    if args.model.endswith('-norm'):
        feats = custom_layers.L2Normalize(name='feats')(feat_maps)
        feats2 = custom_layers.L2Normalize(name='feats2')(feat_maps2)
    else:
        feats = layers.Activation('linear', name='feats')(feat_maps)
        feats2 = layers.Activation('linear', name='feats2')(feat_maps2)

    # Projected Features
    if args.model.endswith('-norm'):
        proj_feats = layers.Dense(128, kernel_regularizer=regularizer, bias_regularizer=regularizer)(feats)
        proj_feats2 = layers.Dense(128, kernel_regularizer=regularizer, bias_regularizer=regularizer)(feats2)
        proj_feats = custom_layers.L2Normalize(name='projection')(proj_feats)
        proj_feats2 = custom_layers.L2Normalize(name='projection2')(proj_feats2)
    else:
        proj_feats = layers.Dense(128, name='projection', kernel_regularizer=regularizer,
                                  bias_regularizer=regularizer)(feats)
        proj_feats2 = layers.Dense(128, name='projection2', kernel_regularizer=regularizer,
                                   bias_regularizer=regularizer)(feats2)

    # Feature views
    proj_views = custom_layers.FeatViews(name='contrast', dtype=tf.float32)((proj_feats, proj_feats2))

    # Stop gradient at features?
    if args.loss != 'ce':
        feats = tf.stop_gradient(feats)

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
