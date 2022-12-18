
# Experiment reproduction

## Datasets

There are some datasets with automatic download others that do not have it because you have to ask for firewall pass.

## Test sets

I have not been able to handle automatic download for these datasets 
Then, download zips from https://alex04072000.github.io/SingleHDR/ and place them in main folder:

+ Testing data (HDR-Real)
+ Testing data (HDR-Eye)

They can also be found at ![this folder](H:\Projects\AIDI\GN1_AI_Driven_Game_Experience\HDR_datasets)

### Download and organize

Execute `python create_datasets.py -h` to see how to create all the datasets. 

First, automatic datasets:

+ `python manage_datasets.py --organize`
    + Will download the datasets and place them in different folders
+ `python manage_datasets.py --build`
    + Will arrange training and validation. Will create the tfrecords necessary for consumption

### Training and validation

To perform inference train or validate go to `experiments\mqnet`