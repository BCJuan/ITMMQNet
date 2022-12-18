"""For checking functionalities.
Dirty version
"""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import tensorflow as tf
from hdrml.networks.network import WholeModel
from hdrml.experiment import Experiment

# Experiment config
tf.random.set_seed(42)
number = "third_wobranches_like29_reduced_2"
name = "branchnet"
input_shape = (256, 256, 3)
input_shape_prediction = (768, 1280, 3)

# Load Net
whole = WholeModel(shape=input_shape)
experiment = Experiment(name, number, input_shape, config_file="config.ini")
experiment.load_datasets()
experiment.enter_model(model=whole)
experiment.load_custom_trainer()
experiment.train_or_resume(train=False, resume=True)
experiment.convert()
new_model = WholeModel(shape=input_shape_prediction)
_ = new_model(tf.expand_dims(tf.random.uniform(input_shape_prediction, dtype=tf.float32), 0))
new_model.set_weights(experiment.trainer.model.get_weights())
experiment.enter_model(new_model)
experiment.load_custom_trainer()
experiment.hdr_converter.input_shape = input_shape_prediction
input_directory = "<your_input_directory"
experiment.hdr_converter.predict(input_directory, (input_shape_prediction[0], input_shape_prediction[1]), 
                                    experiment.tflite_name_fusion, crop=False, video=False)
 