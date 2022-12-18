import argparse
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gts", type=str)
    parser.add_argument("--preds", type=str)
    return parser.parse_args()


def test_hdreye(folder_gt, folder_pred):
    psnr_values = []
    ssim_values = []
    for subfolder in os.listdir(folder_gt):
        import pdb; pdb.set_trace()
        subfolder_name = os.path.join(folder_gt, subfolder)
        if os.path.isdir(subfolder_name):
            for label_name in tqdm(os.listdir(subfolder_name), total=len(os.listdir(subfolder_name))):
                label_path = os.path.join(subfolder_name, label_name)
                label = tf.io.read_file(label_path)
                label = tf.image.decode_jpeg(label, channels=3)
                image_path = os.path.join(folder_pred, subfolder, label_name)
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                psnr = tf.image.psnr(label, image, 255)
                ssim = tf.image.ssim(label, image, 255)
                # print(label_path, image_path, ssim, psnr)
                psnr_values.append(psnr)
                ssim_values.append(ssim)
    print("PSNR Results {} +- {}".format(np.mean(psnr_values), np.std(psnr_values)))
    print("SSIM Results {} +- {}".format(np.mean(ssim_values), np.std(ssim_values)))


if __name__ == "__main__":
    args = parse()
    gts_folder = os.path.join(args.gts)
    preds_folder = os.path.join(args.preds)
    test_hdreye(gts_folder, preds_folder)
