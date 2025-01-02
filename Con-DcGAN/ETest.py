"""
Script for ENGINE: Enhancing Neuroimaging and Genetic Information by Neural Embedding framework
Written in Tensorflow 2.1.0
"""

# Import APIs
import tensorflow as tf
import numpy as np

class engine(tf.keras.Model):
    tf.keras.backend.set_floatx('float32') #设置默认的浮点类型
    """ENGINE framework"""

    def __init__(self, N_o):
        super(engine, self).__init__()
        self.N_o = N_o # the number of classification outputs
        print (self.N_o)
        print ('------------------------------')