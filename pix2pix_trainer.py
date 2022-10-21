'''
This code is an implementation of the pix2pix model developed by Zhu et. al:

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}
'''

#import modules
import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

#hyperparameters
# Buffer size
BUFFER_SIZE = 50000
# Batch size 
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
# Gray-scale images
OUTPUT_CHANNELS = 1
# Lamda
LAMBDA = 100

# Defining and creating directories if needed
checkpoint_dir = 'pix2pix_training/training_checkpoints'
log_dir = 'pix2pix_training/logs/'
result_dir = 'pix2pix_training/results'
overall_path = 'pix2pix_training'
paths = [overall_path, checkpoint_dir, log_dir, result_dir]
for path in paths:
  isExist = os.path.exists(path)
  if not isExist:
    os.makedirs(path)

#defining helper functions

def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)
  

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 1])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

print('loading dataset')



test_images_folder = "/global/D1/homes/hanne/2D_Face_generating/dataset/image_dataset/test/"
train_images_folder = "/global/D1/homes/hanne/2D_Face_generating/dataset/image_dataset/train/"
val_images_folder = "/global/D1/homes/hanne/2D_Face_generating/dataset/image_dataset/val/"


test_images = os.listdir(test_images_folder)
train_images = os.listdir(train_images_folder)
val_images = os.listdir(val_images_folder)

print('number of test images: ' + str(len(test_images)))
print('number of train images: ' + str(len(train_images)))
print('number of validation images: ' + str(len(val_images)))

