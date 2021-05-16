import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils

# print Tensorflow and numpy versions
print('Tensorflow version: ' + tf.version.VERSION) 
print('numpy version: ' + np.__version__)

filename = 'images/airport.jpeg' # load image from directory

mobile = tf.keras.applications.mobilenet_v2.MobileNetV2() # download and load pre-trained deep learning model with weights

mobile = tf.keras.applications.DenseNet169() #alternative, seemingly more accurate pre-trained deep learning model

# Image pre-processing stage
img = image.load_img(filename, target_size=(224, 224))
resized_img = image.img_to_array(img)
final_image = np.expand_dims(resized_img, axis=0) # adds fourth dimension to three-dimensional RGB image array
final_image = tf.keras.applications.mobilenet.preprocess_input(final_image) 

# Perform predictions
predictions = mobile.predict(final_image)
results = imagenet_utils.decode_predictions(predictions)
print(results)
plt.imshow(img)
plt.show()

