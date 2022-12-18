import configparser

DEFAULT_CONFIG = u"""
[HyperParameters]
lr=0.00005
batch_size=4
epochs=310
initial_epoch=0
validation_freq=5
finetune_freq=5
finetune_unfreeze=0.05
steps_lr=8124
"""

def read_configuration(config_file=None):
    config = configparser.ConfigParser()
    if config_file:
        config.read(config_file)
    else:
        config.read_string(DEFAULT_CONFIG)
    return config

def update_configuration(config, config_file):
    new_config = configparser.ConfigParser()
    new_config.read(config_file)
    for section in new_config.sections():
        for name, item in dict(new_config.items(section)).items():
            config.set(section, name, item)
    return config