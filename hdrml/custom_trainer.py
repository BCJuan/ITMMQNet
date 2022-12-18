import time
import os
import abc
from tqdm import tqdm 
import tensorflow as tf
from pathlib import Path
import numpy as np
from .losses import TotalLoss
from .metrics import PSNR, SSIM
from .config_reader import read_configuration, update_configuration
from .imio import log, read_arrange, arrange_save


class CustomTrainer(abc.ABC):

    def __init__(self,name: str, model: tf.keras.models.Model, 
                 checkpoint_prefix: str, logs_folder: str = "./results/logs",
                 config_file=None):
        self.config = read_configuration()
        if config_file:
            self.config = update_configuration(self.config, config_file)
        self.name = name
        self.model = model
        self.logs_folder = logs_folder
        self.checkpoint_prefix = checkpoint_prefix
        self.writer = tf.summary.create_file_writer(self.logs_folder)
        self.ssim = SSIM()
        self.psnr = PSNR()
        self.l1_loss = TotalLoss()
        steps_per_epoch = self.config.getint("HyperParameters", "steps_lr")
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
                            self.config.getfloat("HyperParameters", "lr"), steps_per_epoch, 0.99,
                            staircase=True, name=None
                )
        self.g_opt = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)
        self.reset_losses_metrics()
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.g_opt,
                                              generator=self.model)
        self.pct = self.config.getfloat("HyperParameters", "finetune_unfreeze")

    def reset_losses_metrics(self):
        self.loss_l1_values = 0
        self.loss_perceptual_values = 0
        self.loss_l2_values = 0
        self.total_loss = 0
        self.csim_loss = 0    
        self.ssim.reset_states()
        self.psnr.reset_states()

    def add_losses(self, l1, l2, csim, perceptual):
        self.loss_l1_values += l1
        self.loss_l2_values += l2
        self.loss_perceptual_values +=  perceptual
        self.csim_loss += csim
        self.total_loss += l2 + l1 + csim + perceptual

    @tf.function
    def train_step(self, batch_X, batch_y):
        with tf.GradientTape() as tape:
            y_g_pred = self.model(batch_X, training=True)
            l2, l1, perceptual, csim = self.l1_loss.call(batch_y, y_g_pred, batch_X)
            t_loss = l2 + l1 + csim + perceptual 
        grads = tape.gradient(t_loss, self.model.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.model.trainable_weights))
        return l2, l1,  perceptual, csim

    @tf.function
    def eval_step(self, batch_X, batch_y):
        y_pred = self.model(batch_X, training=False)
        l2,  l1,  perceptual,  csim = self.l1_loss.call(batch_y, y_pred, batch_X)
        self.psnr.update_state(batch_y, y_pred)
        self.ssim.update_state(batch_y, y_pred)
        return l2, l1, perceptual, csim

    def train(self, train_set, val_set):
        best_loss = np.inf
        for i in tqdm(range(
            self.config.getint("HyperParameters", "initial_epoch"),
            self.config.getint("HyperParameters", "initial_epoch") + self.config.getint("HyperParameters", "epochs"))):
            step = 0

            for step, (batch_X, batch_y) in enumerate(train_set):
                l2, l1, perceptual, csim = self.train_step(batch_X, batch_y)
                self.add_losses(l1, l2, csim, perceptual)
            self.log_losses("train", step + 1, i)
            self.reset_losses_metrics()

            if i % self.config.getint("HyperParameters", "validation_freq") == 0:
                for step, (batch_X, batch_y) in enumerate(val_set):
                    l2, l1,  perceptual, csim = self.eval_step(batch_X, batch_y)
                    self.add_losses(l1, l2, csim, perceptual)
                self.log_losses("val", step + 1, i)
                self.log_metrics("val", i)
                if (self.total_loss )/(step + 1) < best_loss:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    self.model.save(self.checkpoint_prefix)
                    best_loss = (self.total_loss)/(step + 1)
                self.reset_losses_metrics()
            
            if i % self.config.getint("HyperParameters", "finetune_freq") == 0 and i !=0:
                self.model.net.get_layer("down_stack").trainable = True
                n_layers = round(len(self.model.net.get_layer("down_stack").layers)*(1-self.pct))
                for j, layer in enumerate(self.model.net.get_layer("down_stack").layers):
                    if j < n_layers or type(layer) == tf.keras.layers.BatchNormalization:
                        layer.trainable = False
                self.pct += self.config.getfloat("HyperParameters", "finetune_unfreeze")

    def log_losses(self, name, number, epoch):
        with self.writer.as_default():
            tf.summary.scalar('gen_l1_loss' + "_" + name, self.loss_l1_values/number, step=epoch)
            tf.summary.scalar('gen_l2_loss' + "_" + name, self.loss_l2_values/number, step=epoch)
            tf.summary.scalar('gen_perceptual_loss' + "_" + name, self.loss_perceptual_values/number, step=epoch)
            tf.summary.scalar('gen_csim_loss' + "_" + name, self.csim_loss/number, step=epoch)
            tf.summary.scalar('gen_total_loss' + "_" + name, self.total_loss/number, step=epoch)
            tf.summary.scalar('Learning rate' + "_" + name, self.g_opt.learning_rate(epoch*number + number), step=epoch)

    def log_metrics(self, name, epoch):
        with self.writer.as_default():
            tf.summary.scalar('psnr'+ "_"+ name, self.psnr.result(), step=epoch)
            tf.summary.scalar('ssim'+ "_"  + name, self.ssim.result(), step=epoch)

    def resume(self):
        directory = Path(self.checkpoint_prefix).parent
        self.checkpoint.restore(tf.train.latest_checkpoint(directory))
        if sum([i.endswith(".pb") for i in os.listdir(self.checkpoint_prefix)]) > 0:
            self.model = tf.keras.models.load_model(
                self.checkpoint_prefix,
                custom_objects={"TotalLoss": TotalLoss()})

    def predict(self, folder, shape, crop):
        for image_name in os.listdir(folder):
            pathname = os.path.join(folder, image_name)
            if os.path.isfile(pathname):
                image = read_arrange(pathname, shape, crop=crop)
                output = self.model.predict(tf.expand_dims(image, 0))
                arrange_save(output, pathname)

    def compute_latency(self, ds, num_runs=150, warm_up=1):
        latencies = []
        counter = 0
        for images, _ in ds:
            time_init = time.time()
            _ = self.model.predict_on_batch(images)
            time_final = time.time() - time_init
            if counter >= warm_up:
                latencies.append(time_final/images.shape[0])
            if counter >num_runs:
                break
            counter += images.shape[0]
        with self.writer.as_default():
            tf.summary.scalar("GPU Latency", np.mean(latencies), step=0)

    def evaluate(self, test_set, name, rounds=5):
        psnr_r = []
        ssim_r = []
        self.model.trainable = False
        for _ in range(rounds):
            self.reset_losses_metrics()
            for images, labels in test_set:
                preds = self.model(images, training=False)
                self.psnr.update_state(labels, preds)
                self.ssim.update_state(labels, preds)
            psnr_r.append(self.psnr.result())
            ssim_r.append(self.ssim.result())
        self.log_metrics("Test" + name, 0)
        with self.writer.as_default():
            tf.summary.scalar("N Params", self.model.count_params(), step=0)
        psnr_r = np.mean(psnr_r, axis=0)
        ssim_r = np.mean(ssim_r, axis=0)
        log.info("Test result on {} test:  {} ".format(name, psnr_r))
        log.info("Test result on {} test:  {} ".format(name, ssim_r))
        log.info("Total number of params: {}".format(self.model.count_params()))