"""import streamlit as st
st.write('hello paul')
image_file = st.file_uploader("Image",type = ['jpg','jpeg','png'])
st.image(image_file)"""
import tensorflow as tf
import tensorflow_hub as tfhub
import os
import numpy as np
import pandas as pd

def load_model(model_path):
  print(f'Loading model saved to {model_path}')
  model = tf.keras.models.load_model(model_path,custom_objects = {'KerasLayer':tfhub.KerasLayer})
  print(f'Successfully loaded model saved to {model_path}')
  return model
model_path = "C:/Users/ACER NITRO/Desktop/dogify/dog_app/dogify/models/20230330-12441680180287full-image-set-imagenetv3-Adam-lr-0.0002final.h5"
model = load_model(model_path)
my_dog_img_path = 'C:/Users/ACER NITRO/Desktop/dogify/dog_app/dogify/my_dog_images/'
my_dog_img_paths = [my_dog_img_path + name for name in os.listdir(my_dog_img_path)]
img_size = 224
batch_size = 32
def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image,channels = 3)
  image = tf.image.convert_image_dtype(image,tf.float32)
  image = tf.image.resize(image,size =(img_size,img_size))
  return image
def get_image_label(image_path,label):
  image = process_image(image_path=image_path)
  return image,label
def augmentation_function(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_rotation(image, 0.15)
    return image
def create_data_batches(X,y = None,batch_size = batch_size,valid_data = False,test_data = False,data_augmentation = False):
  if test_data:
    print('creating test data batches...')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
    data_batch = data.map(process_image).batch(batch_size)
    return data_batch
  elif valid_data:
    print('creating validation data batches...')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data_batch = data.map(get_image_label).batch(batch_size)
    return data_batch
  else:
    print('creating training data batches...')
    data = tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
    data = data.shuffle(buffer_size = len(X))
    data = data.map(get_image_label)
    if data_augmentation:
      data_batch = data.map(augmentation_function)
    data_batch = data.batch(batch_size)
    return data_batch
def get_predicted_label(pred_prob):
  return unique_labels[np.argmax(pred_prob)]
def get_unique_breeds(path):
    label_csv = pd.read_csv(path)
    labels = np.array(label_csv['breed'])
    unique_labels = np.unique(labels)
    return unique_labels
labels_path = 'C:/Users/ACER NITRO/computer vision/computer vision udemy/dogify/labels.csv'
unique_labels = get_unique_breeds(labels_path)
if __name__ == '__main__':
  my_dog_data = create_data_batches(my_dog_img_paths,test_data = True)
  my_dog_pred = model.predict(my_dog_data)
  my_dog_pred_labels = [get_predicted_label(my_dog_pred[i]) for i in range(len(my_dog_pred))]
  print(my_dog_pred_labels)