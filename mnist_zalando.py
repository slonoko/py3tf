import os
import time
import tensorflow as tf
import numpy as np

LABEL_DIMENSION = 10

tf.keras.datasets.fashion_mnist.load_data()
