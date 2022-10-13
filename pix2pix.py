'''
This code is an implementation of the pix2pix developed by Zhu et. al:

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}

The model is trained on a custom made dataset consisting of 2D images of human faces and the corresponding sketches. 
The code generates a 2D image using a random sample from images in the test set. The output is stored under the output folder and named 2D_image_grayscale.jpg. 
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
# The training set consists of 50000 images. Batch size set as recommended by the pix2pix paper
BUFFER_SIZE = 50000
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256
# The images is gray-scale
OUTPUT_CHANNELS = 1
# Labmda is set as recommended by the pix2pix paper
LAMBDA = 100
# location of pre-trained model
checkpoint_dir = '/content/gdrive/MyDrive/Colab Notebooks/ACIT/4030 - Machine learning for images and 3D data/Project/2D_Face_generating/training_checkpoints/'
# location to store output
result_dir = 'output'

# function to load images
def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real human face 
  # - one with a sketch of a human face 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image
 
# function to resize image, ensuring image is of correct size  
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

# Function to load test images 
def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

# path to folder containing some test images
path_dataset = "dataset"
PATH  = pathlib.Path(path_dataset)

# loading the test dataset
try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / '*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / '*.jpg'))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

# down sampler model
def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                             kernel_initializer=initializer, use_bias=False)
  )

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result 

# Defining upsampler model
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', 
                             kernel_initializer=initializer, use_bias=False)
  )

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result 
  
# defining generator model 
def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 1])

  down_stack = [
      downsample(64, 4, apply_batchnorm=False), # (batch size, 128, 128, 64)
      downsample(128, 4),                       # (batch_size, 64, 64, 128)
      downsample(256, 4),                       # (batch_size, 32, 32, 256)
      downsample(512, 4),                       # (batch_size, 16, 16, 512)
      downsample(512, 4),                       # (batch_size, 8, 8, 512)
      downsample(512, 4),                       # (batch_size, 4, 4, 512)
      downsample(512, 4),                       # (batch_size, 2, 2, 513)
      downsample(512, 4),                       # (batch_size, 1, 1, 512)
  ]

  up_stack = [
      upsample(512, 4, apply_dropout=True),     # (batch_size, 2, 2, 1024)
      upsample(512, 4, apply_dropout=True),     # (batch_size, 4, 4, 1024)
      upsample(512, 4, apply_dropout=True),     # (batch_size_ 8,8, 1024)
      upsample(512, 4),                         # (batch_size, 16, 16, 1034)
      upsample(256, 4),                         # (batch_size, 32, 32, 512)
      upsample(128, 4),                         # (batch_size, 64, 64, 256)
      upsample(64, 4),                          # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0.0, 0.02)
  last = tf.keras.layers.Conv2DTranspose(
      OUTPUT_CHANNELS, 4, strides=2, padding='same',
      kernel_initializer=initializer, activation='tanh'   # (batch_size, 256, 256, 3)
  )

  x = inputs 

  # downsampling through the model 
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  #upsampling and establishing the skip connections 
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])
  
  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)
  
# creating generator model 
generator = Generator()

# defining discriminator model 
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 1], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)
  
# creating discriminator model 
discriminator = Discriminator()

# defining optimizers and a checkpoint saver
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, 
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

# Defining function to generate immages 
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['input image', 'ground truth', 'predicted image']

  for i in range(3):
    plt.subplot(1,3,i+1)
    plt.title(title[i])
    # getting pixel values in the [0,1] range to plot 
    plt.imshow(display_list[i][:,:,0] * 0.5 + 0.5, cmap='gray')
    plt.axis('off')

  # saving the generated together with the input, for comparison
  name = result_dir + '/pix2pix_result.jpg'
  plt.savefig(name)
  plt.close()

  # extracting the generated iamge
  generated = prediction[0][:,:,0] 
  
  # saving the generated image
  plt.imshow(generated * 0.5 * 0.5, cmap='gray')
  plt.axis('off')
  name = result_dir + '/2D_image_grayscale.jpg'
  plt.savefig(name, bbox_inches='tight')
  
#restoring the latest checkpoing in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# generating an image
for inp, tar in test_dataset.take(1):
  generate_images(generator, inp, tar)

