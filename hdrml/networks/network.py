import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.layers import VersionAwareLayers
from .blocks import DownBlock, _inverted_res_block

layers = VersionAwareLayers()

def BranchNet(input_shape=(256, 256, 3), skips=None):
    ins = tf.keras.Input(shape=input_shape)
    layer_names = [
        "block_1_expand_relu", # 128, 128
        "block_3_expand_relu", # 64, 64
        "block_6_expand_relu", # 32, 32
        "block_13_expand_relu", # 16, 16
        "block_16_project_BN"
    ]
    model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, alpha=0.35)
    outlayers = [model.get_layer(name).output for name in layer_names]
    out = ins*255
    out = tf.keras.applications.mobilenet.preprocess_input(out)
    down_stack = tf.keras.Model(inputs=model.input, outputs=outlayers, name="down_stack")
    down_stack.trainable = False
    skips = down_stack(out)
    x = _inverted_res_block(skips[-1], filters=32, alpha=1, stride=1, expansion=6, block_id=1, se=False, ca=True, ratio=2)
    x = _inverted_res_block(x, filters=32, alpha=1, stride=1, expansion=6, block_id=2, se=False, ca=False, ratio=2)
    x = layers.UpSampling2D(2)(x)
    x = tf.concat([x, skips[-2]], axis=-1)
    x = _inverted_res_block(x, filters=24, alpha=1, stride=1, expansion=6, block_id=4, se=False, ca=True, ratio=2)
    x = _inverted_res_block(x, filters=24, alpha=1, stride=1, expansion=6, block_id=5, se=False, ca=False, ratio=2)
    x = layers.UpSampling2D(2)(x)
    x = tf.concat([x, skips[-3]], axis=-1)
    x = _inverted_res_block(x, filters=16, alpha=1, stride=1, expansion=6, block_id=7, se=False, ca=True, ratio=2)
    x = layers.UpSampling2D(2)(x)
    x = tf.concat([x, skips[-4]], axis=-1)
    x = _inverted_res_block(x, filters=8, alpha=1, stride=1, expansion=6, block_id=10, se=False, ca=True, ratio=2)
    x = layers.UpSampling2D(2)(x)
    x = tf.concat([x, skips[-5]], axis=-1)
    x = DownBlock(16, 1, 1, 1, "same", False, False)(x)
    x = layers.UpSampling2D(2)(x)
    x = tf.concat([x, ins], axis=-1)
    x = DownBlock(16, 1, 1, 1, "same", False, False)(x)
    return tf.keras.Model(inputs=ins, outputs=x, name="net")


def FusionModelConv(shape):
    return tf.keras.Sequential([
        tf.keras.Input(shape=shape),
        tf.keras.layers.Conv2D(12, 3, 1, dilation_rate=1, padding="same", use_bias=True, kernel_initializer="he_normal"),
        tfa.layers.InstanceNormalization(axis=3, 
                                        center=True, 
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(6, 1, 1, dilation_rate=1, padding="same", use_bias=True, kernel_initializer="he_normal"),
        tfa.layers.InstanceNormalization(axis=3, 
                                        center=True, 
                                        scale=True,
                                        beta_initializer="random_uniform",
                                        gamma_initializer="random_uniform"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(3, 1, activation="tanh", use_bias=False)])


class WholeModel(tf.keras.Model):

    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape
        self.ins = tf.keras.Input(shape=shape)
        self.net = BranchNet(input_shape=shape)
        self.fusion = FusionModelConv(self.net.output_shape[1:])

    def call(self, X):
        out = self.net(X)
        out1 = self.fusion(out)
        return tf.nn.sigmoid(tf.add(X, out1))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "shape": self.shape}
