import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras import backend
from tensorflow.python.keras.applications import imagenet_utils


class DownBlock(tf.keras.layers.Layer):

    def __init__(self, n, k, s, d, p, b, pool, **kwargs):
        super().__init__(**kwargs)
        self.n = n
        self.k = k
        self.s = s
        self.d = d
        self.p = p
        self.b = b
        self.conv1 = tf.keras.layers.Conv2D(self.n, self.k, self.s, dilation_rate=self.d, padding=self.p, use_bias=self.b, kernel_initializer="he_normal")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.pool = pool
        self.pool_op = tf.keras.layers.MaxPool2D(2)

    def call(self, X):
        out = X
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        if self.pool:
            out = self.pool_op(out)
        return out

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "n": self.n, "k": self.k, "s": self.s, "d": self.d, "p": self.p, "b": self.b, "pool": self.pool}


# https://stackoverflow.com/questions/39368367/tf-nn-depthwise-conv2d-is-too-slow-is-it-normal
# https://arxiv.org/pdf/1803.09926.pdf
# https://towardsdatascience.com/mobilenetv2-inverted-residuals-and-linear-bottlenecks-8a4362f4ffd5
# https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
# https://github.com/tensorflow/tensorflow/blob/fcc4b966f1265f466e82617020af93670141b009/tensorflow/python/keras/applications/mobilenet_v2.py#L417

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, se, ca, ratio):
    """Inverted ResNet block."""
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    if block_id:
        # Expand
        x = layers.Conv2D(
            expansion * in_channels,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + 'expand')(
                x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand_BN')(
                x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, 3),
            name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size=3,
        strides=stride,
        activation=None,
        use_bias=False,
        padding='same' if stride == 1 else 'valid',
        name=prefix + 'depthwise')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise_BN')(
            x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(
        pointwise_filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'project')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project_BN')(
            x)
    if se:
        x = squeeze_excite_block(x, ratio=ratio)
    elif ca:
        x = channel_attention_block(x, ratio=ratio)
    if in_channels == pointwise_filters and stride == 1:
        return layers.Add(name=prefix + 'add')([inputs, x])
    return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# https://www.machinecurve.com/index.php/2019/09/24/creating-depthwise-separable-convolutions-in-keras/
# https://github.com/titu1994/keras-squeeze-excite-network
def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = backend.int_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = tf.math.reduce_mean(init, axis=[1, 2])
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if backend.image_data_format() == 'channels_first':
        se = layers.Permute((3, 1, 2))(se)

    x = layers.multiply([init, se])
    return x


# https://github.com/yulunzhang/RCAN/blob/master/RCAN_TrainCode/code/model/rcan.py
def channel_attention_block(tensor, ratio=16):
    init = tensor
    channel_axis = -1
    filters = backend.int_shape(init)[channel_axis]
    se_shape = (1, 1, filters)

    se = tf.reduce_mean(init, axis=[1, 2], keepdims=True)
    se = layers.Conv2D(filters // ratio, 1, activation='relu', kernel_initializer='he_normal', use_bias=True)(se)
    se = layers.Conv2D(filters, 1, activation='sigmoid', kernel_initializer='he_normal', use_bias=True)(se)

    x = layers.multiply([init, se])
    return x


def spatial_attention_block(tensor, ratio=16):
    init = tensor
    se = layers.Conv2D(1, 1, activation="sigmoid", kernel_initializer='he_normal', use_bias=True)(init)
    x = layers.multiply([init, se])
    return x


def channel_spatial_attention_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
    filters = backend.int_shape(init)[channel_axis]
    se = layers.DepthwiseConv2D(filters, 1, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, padding='same')(init)
    se1 = layers.multiply([init, se])
    se = layers.Conv2D(1, 1, activation="sigmoid", kernel_initializer='he_normal', use_bias=True)(init)
    x = layers.multiply([se1, se])
    return x

