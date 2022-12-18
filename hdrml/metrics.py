import tensorflow as tf
from .imio import tf_tonemap_test, tf_interpolate


class SSIM(tf.keras.metrics.Metric):

    def __init__(self, max_value=1.0,  **kwargs):
        super().__init__(**kwargs)
        self.max_value = max_value
        self.value = self.add_weight("value", initializer="zeros")
        self.count = self.add_weight("value", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_pred, y_true = tf_tonemap_test(y_pred, y_true)
        ssim = tf.image.ssim(y_true, y_pred, max_val=self.max_value)
        self.value.assign_add(tf.reduce_sum(ssim))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.value / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "max_value": self.max_value}

    def reset_states(self):
        self.value.assign(0.0)
        self.count.assign(0.0)


class PSNR(tf.keras.metrics.Metric):

    def __init__(self, max_value=1.0, **kwargs):
        super().__init__(**kwargs)
        self.max_value = max_value
        self.value = self.add_weight("value", initializer="zeros")
        self.count = self.add_weight("value", initializer="zeros")

    def update_state(self, y_true, y_pred):
        y_pred, y_true = tf_tonemap_test(y_pred, y_true)
        psnr = tf.image.psnr(y_true, y_pred, max_val=self.max_value)
        self.value.assign_add(tf.reduce_sum(psnr))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.value / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "max_value": self.max_value}

    def reset_states(self):
        self.value.assign(0.0)
        self.count.assign(0.0)