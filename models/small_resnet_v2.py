from models import small_resnet


def SmallResNet50V2(
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the ResNet50V2 architecture."""

    def stack_fn(x):
        x = small_resnet.stack2(x, 16, 3, stride1=1, name='conv2')
        x = small_resnet.stack2(x, 32, 4, stride1=1, name='conv3')
        x = small_resnet.stack2(x, 64, 6, stride1=2, name='conv4')
        return small_resnet.stack2(x, 128, 3, stride1=2, name='conv5')

    return small_resnet.SmallResNet(
        stack_fn,
        True,
        True,
        'resnet50v2',
        include_top,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)


def SmallResNet101V2(
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the ResNet101V2 architecture."""

    def stack_fn(x):
        x = small_resnet.stack2(x, 16, 3, name='conv2')
        x = small_resnet.stack2(x, 32, 4, name='conv3')
        x = small_resnet.stack2(x, 64, 23, name='conv4')
        return small_resnet.stack2(x, 128, 3, stride1=1, name='conv5')

    return small_resnet.SmallResNet(
        stack_fn,
        True,
        True,
        'resnet101v2',
        include_top,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)


def SmallResNet152V2(
        include_top=True,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the ResNet152V2 architecture."""

    def stack_fn(x):
        x = small_resnet.stack2(x, 16, 3, name='conv2')
        x = small_resnet.stack2(x, 32, 8, name='conv3')
        x = small_resnet.stack2(x, 64, 36, name='conv4')
        return small_resnet.stack2(x, 128, 3, stride1=1, name='conv5')

    return small_resnet.SmallResNet(
        stack_fn,
        True,
        True,
        'resnet152v2',
        include_top,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)
