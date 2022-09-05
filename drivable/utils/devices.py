import tensorflow as tf


def initialize_device():
    return (
        tf.distribute.MirroredStrategy()
        if len(tf.config.list_physical_devices("GPU")) > 1
        else tf.distribute.OneDeviceStrategy(device="/gpu:0")
    )
