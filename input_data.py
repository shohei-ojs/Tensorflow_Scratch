import os
import sys
import random
import numpy as np
from PIL import Image


class InputData(object):

  def __init__(self, data_dir='./data'
    , width=32, height=32, depth=3
    , label_size=1, batch_size=64):
    self.width = width
    self.height = height
    self.depth = depth
    self.label_size = label_size
    self.train_data, test_data = self.__read_data(data_dir)
    self.test_labels = [self.__one_hot_vector(label[0]) for (label, image) in test_data]
    self.test_images = [image for (label, image) in test_data]
    self.batch_size = batch_size
    self.idx = 0


  def __read_data(self, data_dir='./data'):
    train_data = []
    test_data = []
    for i in range(1, 6):
      for label, image in self.__read_one_data(data_dir + '/data_batch_' + str(i) + '.bin'):
        train_data.append((label, image))
    
    for label, image in self.__read_one_data(data_dir + '/test_batch.bin'):
      test_data.append((label, image))
    return train_data, test_data

  
  def __read_one_data(self, filepath):
    with open(filepath, 'rb') as f:
        for i in range(10000):
          label = np.frombuffer(f.read(self.label_size), dtype=np.uint8)
          image = np.frombuffer(f.read(self.width*self.height*self.depth), dtype=np.int8) / 255.0
          yield (label, image)
  
  
  def next_batch(self):
    if self.idx + self.batch_size > len(self.train_data):
      self.idx = 0
      random.shuffle(self.train_data)
    batch = self.train_data[self.idx:self.idx+self.batch_size]
    labels = [self.__one_hot_vector(label[0]) for (label, image) in batch]
    images = [image for (label, image) in batch]
    self.idx += self.batch_size
    return labels, images


  def __one_hot_vector(self, index):
    one_hot = np.zeros(10)
    one_hot[index] = 1
    return one_hot
  

  def test_data(self):
    return self.test_labels, self.test_images


def read_image(image_path, image_width=32, image_height=32):
  img = Image.open(image_path)
  size = img.size
  if(size[0]>size[1]):
      diff = size[0] - size[1]
      img = img.crop((diff/2,0,diff/2.0 + size[1],size[1]))
  else:
      diff = size[1] - size[0]
      img = img.crop((0,diff/2.0,size[0],diff/2 + size[0]))


  img.thumbnail((image_width,image_height), Image.ANTIALIAS)
  img = img.convert('RGB')
  if img.size[0] != image_width or img.size[1] != image_height:
      raise ValueError('Image can not be resized.')
  img  = img.resize((image_width, image_height))
  return np.array(img).flatten()