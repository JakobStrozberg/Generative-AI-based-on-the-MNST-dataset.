#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 21:54:51 2023

@author: jakobstrozberg
"""


# Script for loading the trained generator and generating an image

import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
  
# Load the previously saved generator model
new_generator = load_model('generator_model.h5')

# Generate a random noise vector
noise = tf.random.normal([1, 100])

# Use the generator model to create a fake image
generated_image = new_generator(noise, training=False)

plt.imshow((generated_image[0, :, :, 0] + 1) / 2, cmap='gray')
plt.axis('off')
plt.show()

# 