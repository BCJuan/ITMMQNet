import os
import time
import logging
from numpy.random import uniform
import tensorflow as tf
import numpy as np
import cv2


def logger():
    logging.basicConfig(filename=time.strftime("%Y%m%d_%H%M%S") + ".txt", level=logging.INFO, filemode='w')
    return logging


log = logger()


def interpolate(x):
    return np.ndarray.astype(np.interp(x, [np.min(x), np.max(x)], [0., 1.]), np.float32)


@tf.function
def tf_interpolate(inp):
    return tf.numpy_function(interpolate, [inp], tf.float32)


#########################################################################################################
# from https://github.com/dmarnerides/hdr-expandnet/blob/2b47572a88f247e0f44eb14065b0ed4fc9618e0d/util.py
# as well as the other methods
class BaseTMO(object):
    def __call__(self, img):
        return self.op.process(img)


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            scale = uniform(0.65, 0.85)

        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )


class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.0,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Durand(BaseTMO):
    def __init__(
        self,
        contrast=3,
        saturation=1.0,
        sigma_space=8,
        sigma_color=0.4,
        gamma=2.0,
        randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            contrast = uniform(3.5)
        self.op = cv2.xphoto.createTonemapDurand(
            contrast=contrast,
            saturation=saturation,
            sigma_space=sigma_space,
            sigma_color=sigma_color,
            gamma=gamma,
        )


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            bias = uniform(0.7, 0.9)

        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )


class PercentileExposure(object):
    def __init__(self, gamma=2.0, low_perc=10, high_perc=90, randomize=False):
        if randomize:
            gamma = uniform(1.8, 2.2)
            low_perc = uniform(0, 15)
            high_perc = uniform(85, 100)
        self.gamma = gamma
        self.low_perc = low_perc
        self.high_perc = high_perc

    def __call__(self, x):
        low, high = np.percentile(x, (self.low_perc, self.high_perc))
        return interpolate(np.clip(x, low, high)) ** (1 / self.gamma)


TRAIN_TMO_DICT = {
    'exposure': PercentileExposure,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
#    'durand': Durand,
}
##########################################################################

def random_tonemap(img):
    bgr = img[...,::-1]
    tmos = list(TRAIN_TMO_DICT.keys())
    nans = True
    while nans:
        try:
            choice = np.random.randint(0, len(tmos))
            tmo = TRAIN_TMO_DICT[tmos[choice]](randomize=True)
            tm_bgr = tmo(bgr)
            nans = False
        except:
            pass
    img = tm_bgr[...,::-1]
    return np.ndarray.astype(img, np.float32)


@tf.function
def tf_tonemap(inp):
    return tf.numpy_function(random_tonemap, [inp], tf.float32)


def tonemap_test(img, gt):
    bgr = img[...,::-1]
    bgt = gt[...,::-1]
    tmos = list(TRAIN_TMO_DICT.keys())
    nans = True
    while nans:
        try:
            choice = np.random.randint(0, len(tmos))
            tmo = TRAIN_TMO_DICT[tmos[choice]](randomize=True)
            tm_bgr = tmo(bgr)
            tm_gt = tmo(bgt)
            nans = False
        except:
            pass
    img = tm_bgr[...,::-1]
    gt = tm_gt[...,::-1]
    return np.ndarray.astype(img, np.float32), np.ndarray.astype(gt, np.float32)


@tf.function
def tf_tonemap_test(img, gt):
    return tf.numpy_function(tonemap_test, [img, gt], [tf.float32, tf.float32])


def read_image(img_path, size, crop=False):
    image = tf.io.read_file(img_path)
    if os.path.basename(img_path).endswith(".png"):
        image = tf.image.decode_png(image, channels=3, dtype=tf.uint8)
    elif os.path.basename(img_path).endswith(".jpg") or os.path.basename(img_path).endswith(".jpeg"):
        image = tf.image.decode_jpeg(image, channels=3)
    height, width = image.shape[0], image.shape[1]
    if crop:
        if width < height:
            new_size = (width, width)
        else:
            new_size = (height, height)
        image = tf.image.crop_to_bounding_box(
                    image, 0, 0, new_size[0], new_size[1]
                )
    image = tf.image.resize(image, size)
    return image


def read_arrange(image_path, input_shape, crop=False):
    output_dir_inputs = os.path.join(os.path.dirname(image_path), "inputs")
    output_dir_outputs = os.path.join(os.path.dirname(image_path), "outputs")
    check_existence(output_dir_inputs)
    check_existence(output_dir_outputs)
    image = read_image(image_path, input_shape, crop=crop)
    image = tf_interpolate(tf.cast(image, tf.float32))
    image_process = image[..., ::-1].numpy()*255
    cv2.imwrite(os.path.join(output_dir_inputs, os.path.basename(image_path).split(".")[0] + "_resized.png"), image_process)
    return image


def check_existence(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def arrange_save(output, image_path):
    output_dir_outputs = os.path.join(os.path.dirname(image_path), "outputs")
    output = tf_interpolate(output)
    output = tf.squeeze(output)
    output = output[..., ::-1].numpy()
    cv2.imwrite(os.path.join(output_dir_outputs, os.path.basename(image_path).split(".")[0] + ".hdr"), output)


def to_cv2(img):
    img = img[..., ::-1]
    return img


def save_hdr(image, name):
    image = tf_interpolate(image)
    image_cv2 = to_cv2(image.numpy())
    cv2.imwrite(name, image_cv2)

def read_arrange_video(image_path, frame, counter):
    output_dir_inputs = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0], "inputs")
    output_dir_outputs = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0], "outputs")
    check_existence(output_dir_inputs)
    check_existence(output_dir_outputs)
    frame = cv2.resize(frame, (1280, 768))
    frame = tf_interpolate(tf.cast(frame, tf.float32))
    cv2.imwrite(os.path.join(output_dir_inputs,  str(counter) + ".png"), frame.numpy()*255)
    frame = frame[..., ::-1]
    return frame

def arrange_save_video(output, image_path, counter):
    output_dir_outputs = os.path.join(os.path.dirname(image_path), os.path.splitext(os.path.basename(image_path))[0], "outputs")
    output = tf_interpolate(output)
    output = tf.squeeze(output)
    output = output[..., ::-1].numpy()
    cv2.imwrite(os.path.join(output_dir_outputs,  str(counter) + ".hdr"), output)