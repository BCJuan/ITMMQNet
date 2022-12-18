"""
Module for serving data
"""
import random
import os
import tensorflow as tf
import tensorflow_io as tfio
from .tfrecord_builders import OnlyHDRTFRecordBuilder, TFRecordBuilder
from .datasets import ROOT
from ..imio import tf_interpolate, tf_tonemap


class OnlyHDRTFRecordServer(OnlyHDRTFRecordBuilder):

    def __init__(self, folder_name, batch_size, root=ROOT, size_train=(256, 256), augment=True):
        super().__init__(folder_name, root, 2)
        tfrecords = sorted([os.path.join(self.tfrecord_path, i) for i in os.listdir(self.tfrecord_path)])
        random.shuffle(tfrecords)
        self.dataset = tf.data.TFRecordDataset(
            tfrecords,
            compression_type=OnlyHDRTFRecordBuilder.compression_type,
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        self.batch_size = batch_size
        self.size_train = size_train
        self.augment = augment

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.image_feature_description)
    
    def prepare_dataset(self):
        dataset = self.dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.read_images, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sizing, num_parallel_calls=tf.data.AUTOTUNE)
        if self.augment:
            dataset = dataset.map(self.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(16, reshuffle_each_iteration=True)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def read_images(self, image_features):
        ground_truth = tfio.experimental.image.decode_hdr(image_features["ground_truth"])
        return ground_truth

    def augmentation(self, ground):
        do_flip = tf.random.uniform([], dtype=tf.float32) > 0.5
        ground = tf.cond(do_flip, lambda: tf.image.flip_left_right(ground), lambda: ground)
        do_up = tf.random.uniform([], dtype=tf.float32) > 0.5
        ground = tf.cond(do_up, lambda: tf.image.flip_up_down(ground), lambda: ground)
        return ground

    def preprocess(self, ground_truth):
        image_raw = tf.identity(ground_truth)
        ground_truth = tf.cast(tf_interpolate(ground_truth), tf.float32)
        body = lambda j: tf.cast(tf_interpolate(tf_tonemap(image_raw)), tf.float32)
        cond = lambda i: tf.math.is_nan(tf.reduce_max(i))
        image_raw_tone = body(image_raw)
        image_raw_tone = tf.while_loop(cond, body, [image_raw_tone])[0]
        ground_truth.set_shape([None, None, 3])
        image_raw_tone.set_shape([None, None, 3])
        return image_raw_tone, ground_truth

    def sizing(self, ground):
        width = tf.shape(ground)[1]
        height = tf.shape(ground)[0]
        if width < height:
            new_size = (width, width)
            start_h = tf.random.uniform([], minval=0, maxval=height - width - 1, dtype=tf.int32)
            start_w = 0
        else:
            new_size = (height, height)
            if width == height:
                start_w = 0
            else:
                start_w = tf.random.uniform([], minval=0, maxval=width - height, dtype=tf.int32)
            start_h = 0
        ground = tf.image.crop_to_bounding_box(ground, start_h, start_w, new_size[0], new_size[1])
        ground = tf.image.resize(ground, self.size_train)
        return ground


class TFRecordServer(TFRecordBuilder):

    def __init__(self, folder_name, batch_size, size=(256, 256), root=ROOT):
        super().__init__(folder_name, root, 2)
        tfrecords = sorted([os.path.join(self.tfrecord_path, i) for i in os.listdir(self.tfrecord_path)])
        random.shuffle(tfrecords)
        self.dataset = tf.data.TFRecordDataset(
            tfrecords,
            compression_type=OnlyHDRTFRecordBuilder.compression_type,
            num_parallel_reads=tf.data.AUTOTUNE,
        )
        self.batch_size = batch_size
        self.size = size

    def _parse_image_function(self, example_proto):
        return tf.io.parse_single_example(example_proto, self.image_feature_description)

    def prepare_dataset(self):
        dataset = self.dataset.map(self._parse_image_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.read_images, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.sizing, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def read_images(self, image_features):
        ground_truth = tfio.experimental.image.decode_hdr(image_features["ground_truth"])
        image_raw = tf.io.decode_image(image_features["input"], dtype=tf.uint8,
                                expand_animations=False, channels=3)
        return image_raw, ground_truth

    def preprocess(self, image_raw, ground_truth):
        ground_truth = tf.cast(tf_interpolate(ground_truth), tf.float32)
        image_raw = tf_interpolate(tf.cast(image_raw, tf.float32))
        return image_raw, ground_truth

    def sizing(self, image, ground):
        width = tf.shape(ground)[1]
        height = tf.shape(ground)[0]
        if width < height:
            new_size = (width, width)
            start_h = tf.random.uniform([], minval=0, maxval=height - width - 1 , dtype=tf.int32)
            start_w = 0
        else:
            new_size = (height, height)
            if width == height:
                start_w = 0
            else:
                start_w = tf.random.uniform([], minval=0, maxval=width - height, dtype=tf.int32)
            start_h = 0
        ground = tf.image.crop_to_bounding_box(ground, start_h, start_w, new_size[0], new_size[1])
        image = tf.image.crop_to_bounding_box(image, start_h, start_w, new_size[0], new_size[1])
        ground = tf.image.resize(ground, self.size)
        image = tf.image.resize(image, self.size)
        return image, ground
