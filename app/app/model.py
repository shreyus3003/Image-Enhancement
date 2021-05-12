import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import requests
import json
import base64
import argparse
from PIL import Image


os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
IMAGE_PATH = "original.png"
#SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
#SAVED_MODEL_PATH = "http://localhost:8501/v1/models/tfgan:predict"
SAVED_MODEL_PATH = "http://35.224.126.101:8501/v1/models/model:predict"
def preprocess_image(image_path):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(tf.io.read_file(image_path))
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)
def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save to.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.jpg" % filename)
  print("Saved as %s.jpg" % filename)
  
  
def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)

def get_prediction(IMAGE_PATH):
  hr_image = preprocess_image(IMAGE_PATH)
  # Plotting Original Resolution image
  plot_image(tf.squeeze(hr_image), title="Original Image")
  save_image(tf.squeeze(hr_image), filename="Original Image")

  # model = hub.load(SAVED_MODEL_PATH)
  # start = time.time()
  # fake_image = model(hr_image)
  image = Image.open("Original Image.jpg") 
    # convert image to array 
  im =  np.asarray(image) 
    # add the 4th dimension 
  im = np.expand_dims(im, axis=0) 
  #im= im/255
  print(im)
  print("*************************")
  print("Image shape: ",im.shape) 
  data = json.dumps({"instances": im.tolist()})
  rv = requests.post(SAVED_MODEL_PATH, data=data)
  response = json.loads(rv.text)
  response_string = response['predictions'][0]
  
  print(np.asarray(response_string))
  fake_image = tf.image.convert_image_dtype((np.asarray(response_string)), dtype=tf.float32, saturate=True)

  #fake_image = np.array(response_string).astype(float)
  #fake_image = Image.fromarray(np.array(response_string).astype(float))
  #fake_image = Image.fromarray((np.asarray(response_string)* 255).astype('uint8'))
  #out_image.save('rand.png')
  #print(hr_image)
  #fake_image = tf.keras.preprocessing.image.array_to_img(np.asarray(response_string)*255)
  #fake_image = tf.squeeze(fake_image)
  print(tf.reduce_max(fake_image))

  print("######################")

  print(tf.reduce_min(fake_image))



  #fake_image = tf.squeeze(fake_image)
  #print("Time Taken: %f" % (time.time() - start))
 # return(fake_image)
  # Plotting Super Resolution Image
  #plot_image(tf.squeeze(fake_image), title="Super Resolution")
  filename = os.path.join('static', 'SuperResolution')
  #save_image(tf.squeeze(fake_image), filename=filename)
  save_image(fake_image, filename=filename)
