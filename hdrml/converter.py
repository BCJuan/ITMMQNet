"""
Conversion module for preparing tflite file for deployment
"""
import abc
import pathlib
import cv2
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from .metrics import PSNR, SSIM
from .imio import log, arrange_save, read_arrange, save_hdr, read_arrange_video, arrange_save_video


class TFConverter(object):

    def __init__(self, saved_model_dir, tflite_path, fallback=True, rep_dataset=None, input_shape=None):
        self.converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=saved_model_dir)
        self.converter.experimental_new_converter = True
        self.tflite_model = None
        self.tflite_path = pathlib.Path(tflite_path)
        self.dataset = rep_dataset
        self.flag_int = False
        self.interpreter = None
        self.input_shape = input_shape
        self.converter.target_spec.supported_ops = []
        if fallback:
            self.converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)
            self.converter.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)

    def to_tflite(self):
        self.tflite_model = self.converter.convert()

    def to_dynamic(self):
        self.converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        self.tflite_model = self.converter.convert()

    def to_float16(self):
        self.converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        self.converter.target_spec.supported_types = [tf.float16]
        self.tflite_model = self.converter.convert()

    def representative_dataset(self, n_samples=None):
        if n_samples:
            self.dataset = self.dataset.take(n_samples)
        for inputs, _ in self.dataset:
            yield [inputs]

    def to_int8(self, full_int=True, dtype=tf.uint8):
        self.converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        self.converter.representative_dataset = self.representative_dataset
        if full_int:
            self.converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS_INT8)
            self.converter.inference_input_type = dtype
            self.converter.inference_output_type = dtype
        self.tflite_model = self.converter.convert()
        self.flag_int = True

    def to_int16int8(self):
        self.converter.representative_dataset = self.representative_dataset
        self.converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        self.converter.target_spec.supported_ops.append(
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8)
        self.tflite_model = self.converter.convert()

    def save(self):
        self.tflite_path.write_bytes(self.tflite_model)

    @abc.abstractmethod
    def evaluate(self):
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_path))
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_index = self.interpreter.get_output_details()[0]["index"]


class HDRTFConverter(TFConverter):

    def add_interpreters(self, model_path):
        self.interpreter_2 = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter_2.allocate_tensors()
        self.input_index_2 = self.interpreter_2.get_input_details()[0]["index"]
        self.output_index_2 = self.interpreter_2.get_output_details()[0]["index"]

    def evaluate(self, test_ds, model_path, name=None, num_runs=5):
        psnr_mean = []
        ssim_mean = []
        psnr = PSNR()
        ssim = SSIM()
        for i in tqdm(range(num_runs)):
            psnr.reset_states()
            ssim.reset_states()
            for inputs, labels in test_ds:
                for image, label in zip(inputs, labels):
                    label = tf.expand_dims(label, axis=0)
                    output_2 = self.predict_single(image, model_path)
                    output = tf.nn.sigmoid(tf.add(tf.expand_dims(image, 0), output_2))
                    psnr.update_state(label, output)
                    ssim.update_state(label, output)
            psnr_mean.append(psnr.result())
            ssim_mean.append(ssim.result())
        log.info("PSNR Test Value for {} : {}".format(name, np.mean(psnr_mean)))
        log.info("SSIM Test Value for {} :  {}".format(name, np.mean(ssim_mean)))

    def predict(self, folder, shape, model_path, crop, video=False):
        if video:
            for video_name in os.listdir(folder):
                pathname = os.path.join(folder, video_name)
                cap = cv2.VideoCapture(pathname)
                print(pathname)
                counter = 0
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        frame = read_arrange_video(pathname, frame, counter)
                        output_2 = self.predict_single(frame, model_path)
                        output = tf.nn.sigmoid(tf.add(tf.expand_dims(frame, 0), output_2))
                        arrange_save_video(output, pathname, counter)
                        counter += 1
                    else: 
                        break
                cap.release()
                cv2.destroyAllWindows()
        else:
            for image_name in os.listdir(folder):
                pathname = os.path.join(folder, image_name)
                if os.path.isfile(pathname):
                    image = read_arrange(pathname, shape, crop=crop)
                    output_2 = self.predict_single(image, model_path)
                    output = tf.nn.sigmoid(tf.add(tf.expand_dims(image, 0), output_2)) 
                    arrange_save(output, pathname)

    def pred_hdrvdp(self, test_ds, save_path, model_path):
        root_preds = os.path.join(save_path, "preds")
        root_gts = os.path.join(save_path, "gts")
        os.makedirs(root_preds, exist_ok=True)
        os.makedirs(root_gts, exist_ok=True) 
        counter = 0
        for inputs, labels in test_ds:
            for image, label in zip(inputs, labels):
                output_2 = self.predict_single(image, model_path)
                output = tf.nn.sigmoid(tf.add(tf.expand_dims(image, 0), output_2))
                save_hdr(output[0], os.path.join(root_preds, str(counter) + ".hdr"))
                save_hdr(label, os.path.join(root_gts, str(counter) + ".hdr"))
                counter += 1


    def reinitialize_interpreters(self, model_path):
        super().evaluate()
        self.add_interpreters(model_path)
        zero = self.interpreter.get_output_details()[0]['quantization_parameters']['zero_points']
        scales = self.interpreter.get_output_details()[0]['quantization_parameters']['scales']
        zero_in = self.interpreter.get_input_details()[0]['quantization_parameters']['zero_points']
        scales_in = self.interpreter.get_input_details()[0]['quantization_parameters']['scales']
        self.interpreter.resize_tensor_input(self.input_index, (1, *self.input_shape))
        self.interpreter.allocate_tensors()
        output_channels = self.interpreter.get_output_details()[0]['shape'][-1]
        self.interpreter_2.resize_tensor_input(self.input_index_2, (1, self.input_shape[0], self.input_shape[1], output_channels))
        self.interpreter_2.allocate_tensors()
        return zero, scales, zero_in, scales_in

    def predict_single(self, image, model_path):
        zero, scales, zero_in, scales_in = self.reinitialize_interpreters(model_path)
        test_image = tf.cast((image)/scales_in + zero_in, tf.uint8)
        test_image = tf.expand_dims(test_image, axis=0)
        self.interpreter.set_tensor(self.input_index, test_image)
        self.interpreter.invoke()
        output = self.interpreter.tensor(self.output_index)()
        output = (tf.cast(output, tf.float32) - zero) * scales
        self.interpreter_2.set_tensor(
            self.input_index_2, output)
        self.interpreter_2.invoke()
        output_2 = self.interpreter_2.tensor(self.output_index_2)()     
        return output_2
    