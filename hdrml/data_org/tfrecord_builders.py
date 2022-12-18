"""Module for recording data into tfrecords and serving the data afterwards
to a training script
"""

import glob
import os
import shutil
import tensorflow as tf
from tqdm import tqdm
from .datasets import Dataset, ROOT


class OnlyHDRTFRecordBuilder(Dataset):
    """Class for building tfrecord files
    Args:
        folder_name (dir): where the input and output folders are
        root (dir): the parent of folder_name; where all data is
        n_machines (bool): for distribution of data
    """
    compression_type = "GZIP"
    options = tf.io.TFRecordOptions(compression_type=compression_type)

    def __init__(self, folder_name, root=ROOT, n_machines=2):
        super().__init__(folder_name, root=root)
        self.hdr_files = sorted(
            glob.glob(os.path.join(self.ground_truth_folder, "*.*"))
        )
        self.tfrecord_path = os.path.join(self.full_path, "tfrecords")
        n_tfrecords = (n_machines * 10 if (get_size(self.ground_truth_folder) \
            / 1024 / 1024 / (10 * n_machines)) > 100 else 1)
        self.n_files_per_tfrecord = len(os.listdir(self.ground_truth_folder)) // n_tfrecords
        self.image_feature_description = {
            "ground_truth": tf.io.FixedLenFeature([], tf.string),
        }

    def organize(self):
        if os.path.exists(self.tfrecord_path):
            shutil.rmtree(self.tfrecord_path)
            os.mkdir(self.tfrecord_path)
        else:
            os.mkdir(self.tfrecord_path)
        file_count, shard_count = 0, 0
        tfrecord_name = "shard_{}.tfrecords".format(shard_count)
        writer = tf.io.TFRecordWriter(
            os.path.join(self.tfrecord_path, tfrecord_name),
            options=OnlyHDRTFRecordBuilder.options,
        )
        for hdr_file in tqdm(self.hdr_files, total= len(self.hdr_files)):
            if file_count > self.n_files_per_tfrecord:
                shard_count += 1
                tfrecord_name = "shard_{}.tfrecords".format(shard_count)
                writer = tf.io.TFRecordWriter(
                    os.path.join(self.tfrecord_path, tfrecord_name),
                    options=OnlyHDRTFRecordBuilder.options,
                )
                file_count = 0
            writer.write(self.build_example(hdr_file).SerializeToString())
            file_count += 1

    def build_example(self, path_ground_truth):
        """Build a tf.train.Example for writing it to a tfrecord file
        Args:
            path_ground_truth (str): path to the ground truth hdr imag
        Returns:
            tf.train.Example: contains both the input and gt images as features
        """
        ground_string = open(path_ground_truth, "rb").read()

        feature = {
            "ground_truth": _bytes_feature(ground_string),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


class TFRecordBuilder(OnlyHDRTFRecordBuilder):
    """Class for building tfrecord files
    Args:
        folder_name (dir): where the input and output folders are
        root (dir): the parent of folder_name; where all data is
    """
    def __init__(self, folder_name, root=ROOT, n_machines=2):
        super().__init__(folder_name, root=root, n_machines=n_machines)
        self.image_feature_description = {
            "input": tf.io.FixedLenFeature([], tf.string),
            "ground_truth": tf.io.FixedLenFeature([], tf.string),
        }
        self.ldr_files = sorted(glob.glob(os.path.join(self.inputs_folder, "*.*")))

    def organize(self):
        if os.path.exists(self.tfrecord_path):
            shutil.rmtree(self.tfrecord_path)
            os.mkdir(self.tfrecord_path)
        else:
            os.mkdir(self.tfrecord_path)
        file_count, shard_count = 0, 0
        tfrecord_name = "shard_{}.tfrecords".format(shard_count)
        writer = tf.io.TFRecordWriter(
            os.path.join(self.tfrecord_path, tfrecord_name),
            options=OnlyHDRTFRecordBuilder.options,
        )
        for hdr_file, ldr_file in tqdm(
            zip(self.hdr_files, self.ldr_files), total= len(self.hdr_files)):
            if file_count > self.n_files_per_tfrecord:
                shard_count += 1
                tfrecord_name = "shard_{}.tfrecords".format(shard_count)
                writer = tf.io.TFRecordWriter(
                    os.path.join(self.tfrecord_path, tfrecord_name),
                    options=TFRecordBuilder.options,
                )
                file_count = 0
            writer.write(self.build_example(hdr_file, ldr_file).SerializeToString())
            file_count += 1

    def build_example(self, path_ground_truth, path_input_file):
        input_string = open(path_input_file, "rb").read()
        ground_string = open(path_ground_truth, "rb").read()

        feature = {
            "input": _bytes_feature(input_string),
            "ground_truth": _bytes_feature(ground_string),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


def get_size(folder):
    """returns size of folder (only files, not recursive)
    Args:
        folder (dir): folder to measure size
    Returns:
        int: number of bytes in folder (not recursive)
    """
    return sum(
        os.path.getsize(os.path.join(folder, f))
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
    )


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
