# GN1-GAME-HDR-ML-

LDR to HDR Conversion for Smartphones

## Environment

First, `sudo apt-get install openexr` and `sudo apt-get install libopenexr-dev`. Then  follow `https://cloud.google.com/storage/docs/gsutil_install`. Also, `imageio_download_bin freeimage`.

If you work with Anaconda, things will be easier. To have the same environment just: `conda env create --file hdr.yaml -n hdr`

Then do `pip install -e .` here.

## Datasets

For only inference you do not need to do this. Go to experiments, and follow the `README`.

## Inference and training

Go to `experiments\mqnet` and follow `README`. You will need the datasets to perform inference since data is needed for quantization.

