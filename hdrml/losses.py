"""Losses for HDR reconstruction
"""

import tensorflow as tf


class TotalLoss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vgg16 = tf.keras.applications.VGG16(include_top=False, pooling=True, input_shape=(256, 256, 3))
        layers = [
            "block1_pool",
            "block2_pool",
            "block3_pool"
        ]
        output_layers = [vgg16.get_layer(layer_name).output for layer_name in layers]
        self.model = tf.keras.Model(inputs=vgg16.input, outputs=output_layers)

    def call(self, y_true, y_pred, y_input):
        pred_layers = self.model(y_pred)
        gt_layers = self.model(y_true)
        difference = tf.math.reduce_sum([tf.math.reduce_mean(tf.math.reduce_mean(tf.abs(pred1 - pred2), axis=[1, 2, 3])) for pred1, pred2 in zip(pred_layers, gt_layers)])
        csim = 1 - tf.math.reduce_sum(tf.reduce_sum(tf.nn.l2_normalize(y_true)*tf.nn.l2_normalize(y_pred), axis=[1, 2, 3]))
        l1 = tf.reduce_mean(tf.reduce_mean(tf.abs(y_true - y_pred), axis=[1, 2, 3]), axis=0)
        l2 = tf.reduce_mean(tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3]))
        return l2, l1,  0.05 * difference, 0.1 * csim