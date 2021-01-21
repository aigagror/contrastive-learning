import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import applications, layers

from models import custom_layers
from models import small_resnet_v2


def make_model(args, nclass, input_shape):
    input = keras.Input(input_shape)
    input2 = keras.Input(input_shape)

    if args.cnn == 'resnet50v2':
        cnn = applications.ResNet50V2(weights=None, include_top=False)
    elif args.cnn == 'small-resnet50v2':
        cnn = small_resnet_v2.SmallResNet50V2(include_top=False)
    else:
        raise Exception(f'unknown cnn model {args.cnn}')

    stand_img = custom_layers.StandardizeImage()
    avg_pool = layers.GlobalAveragePooling2D()

    feats = avg_pool(cnn(stand_img(input)))
    feats2 = avg_pool(cnn(stand_img(input2)))

    if args.norm_feats:
        feats = custom_layers.L2Normalize()(feats)
        feats2 = custom_layers.L2Normalize()(feats2)

    proj_feats = layers.Dense(128, name='projection')(feats)
    proj_feats2 = layers.Dense(128, name='projection2')(feats2)

    if args.norm_feats:
        proj_feats = custom_layers.L2Normalize()(proj_feats)
        proj_feats2 = custom_layers.L2Normalize()(proj_feats2)

    batch_sims = custom_layers.GlobalBatchSims()((proj_feats, proj_feats2))

    if args.method.startswith('supcon'):
        feats = tf.stop_gradient(feats)
    prediction = layers.Dense(nclass, name='classifier')(feats)

    inputs = {'imgs': input}
    targets = {'labels': prediction}

    if args.method.startswith('supcon'):
        inputs['imgs2'] = input2
        targets['batch_sims'] = batch_sims

    model = keras.Model(inputs, targets)

    # L2 regularization
    regularizer = keras.regularizers.l2(args.l2_reg)
    for module in model.submodules:
        for attr in ['kernel_regularizer', 'bias_regularizer']:
            if hasattr(module, attr):
                setattr(module, attr, regularizer)

    return model
