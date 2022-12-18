
from abc import ABC
import os
import tensorflow as tf
from .config_reader import read_configuration, update_configuration
from .custom_trainer import CustomTrainer
from .data_org.tfrecord_servers import TFRecordServer, OnlyHDRTFRecordServer
from .converter import HDRTFConverter
from .imio import log


class Experiment(ABC):

    def __init__(self, name, number, input_shape, config_file="config.ini", root="../data"):
        # Experiment config
        tf.random.set_seed(42)
        self.name = name
        self.input_shape = input_shape
        self.root = root
        self.number = number
        self.make_folders(name, number)
        self.config_file = config_file
        self.config = read_configuration()
        self.config = update_configuration(self.config, self.config_file)
        self.batch_size = self.config.getint("HyperParameters", "batch_size")

    def make_folders(self, name, number):
        self.folder = os.path.join("./logs", number + "/")
        self.model_folder = os.path.join("./models", number, "model/")
        self.ba_folder = os.path.join("./models", number, "ba/")
        self.fusion_folder = os.path.join("./models", number, "fusion/")
        self.linear_folder = os.path.join("./models", number, "linear/")
        self.tflite_name = os.path.join(self.ba_folder, name + number + ".tflite")
        self.tflite_name_fusion = os.path.join(self.fusion_folder, "fusion_" + number + ".tflite")
        self.tflite_name_linear= os.path.join(self.linear_folder, "linear_" + number + ".tflite")

    def load_datasets(self):
        hdreye = TFRecordServer(folder_name="hdreye_test", batch_size=1, root=self.root, size=self.input_shape[:2])
        self.hdreye_ds = hdreye.prepare_dataset()
        hdrreal = TFRecordServer(folder_name="hdrreal", batch_size=1,
                                 root=self.root, size=self.input_shape[:2])
        self.hdrreal_ds = hdrreal.prepare_dataset()
        train = OnlyHDRTFRecordServer(folder_name="train", batch_size=self.batch_size, root=self.root,
                                      size_train=self.input_shape[:2], augment=True)
        self.train_ds = train.prepare_dataset()
        validation = OnlyHDRTFRecordServer(folder_name="validation", batch_size=self.batch_size, root=self.root,
                                      size_train=self.input_shape[:2], augment=False)
        self.val_ds = validation.prepare_dataset()

    def enter_model(self, model):
        self.model = model

    def load_custom_trainer(self):
        self.trainer = CustomTrainer(name=self.name,
                                    model=self.model, checkpoint_prefix=self.model_folder,
                                    logs_folder=self.folder, config_file=self.config_file)

    def train_or_resume(self, train=True, resume=False):
        self.trainer.model.net.get_layer("down_stack").trainable = train
        if resume:
            self.trainer.resume()
        if train:
            self.trainer.train(self.train_ds, self.val_ds)

    def convert(self):
        self.trainer.model.net.save(self.ba_folder)
        log.info("Number of params BA: {}".format(self.trainer.model.net.count_params()))
        self.hdr_converter = HDRTFConverter(self.ba_folder, self.tflite_name, fallback=False,
                                            rep_dataset=self.val_ds, input_shape=self.input_shape)
        self.hdr_converter.to_int8()
        self.hdr_converter.save()

        self.trainer.model.fusion.save(self.fusion_folder)
        log.info("Number of params Fusion: {}".format(self.trainer.model.fusion.count_params()))
        hdr_converter_fusion = HDRTFConverter(self.fusion_folder, self.tflite_name_fusion,
                                              fallback=True, input_shape=self.input_shape)
        hdr_converter_fusion.to_dynamic()
        hdr_converter_fusion.save()
