from tframe import Classifier
from tframe import mu, tf
from tframe.layers import Activation
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config
from uslp_core import th

import numpy as np

total_filters = []
encode_shape = []

FIRST_FILTER = 5

#region customize layer to U-net
class PadLayerOdd2Even(Layer):
  abbreviation = 'padding'
  full_name = abbreviation

  def __init__(self):
    pass

  @single_input
  def _link(self, x: tf.Tensor):
    output = tf.pad(x, paddings=[[0, 0], [0, x.shape[1] % 2], [0, 0]])
    encode_shape.append(output.shape[1])
    return output

class Crop2Match(Layer):
  abbreviation = 'crop'
  full_name = abbreviation

  def __init__(self, index: int):
    self.index = index

  @single_input
  def _link(self, x: tf.Tensor):
    diff = tf.maximum(0, x.shape[1] - encode_shape[self.index])
    start = diff // 2 + diff % 2
    return x[:, start:start+encode_shape[self.index], :]

# endregion

def init_model(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if flatten: model.add(mu.Flatten())
  return model

def output_and_build(model):
  assert isinstance(model, Classifier)
  assert isinstance(th, Config)
  # Add output layer
  model.add(mu.Conv1D(int(FIRST_FILTER*np.sqrt(2)), 1, activation=Activation('tanh')))
  model.add(mu.AveragePooling1D(th.random_sample_length, th.random_sample_length))
  model.add(mu.Conv1D(th.output_dim, 1, activation=Activation('relu')))
  model.add(mu.Conv1D(th.output_dim, 1, activation=Activation('softmax')))
  # Build model and return
  model.build(metric='accuracy', batch_metric='accuracy')
  return model


def conv1d(filters, kernel_size=9, strides=1):
  """Conv1D layer"""
  return mu.Conv1D(filters, kernel_size, strides,
                   activation=th.activation,
                   use_batchnorm=th.use_batchnorm)


def deconv1d(filters, kernel_size=2, strides=2):
  return mu.HyperDeconv1D(filters, kernel_size, strides)

def maxpool(pool_size=2, strides=2):
  """Maxpool layer"""
  return mu.MaxPool1D(pool_size, strides)

def encoder(filters):
  global total_filters
  encoder = []
  pre_filters = filters
  total_filters.append(pre_filters)
  encoder.append([conv1d(pre_filters), PadLayerOdd2Even()])
  for i in range(11):
    pre_filters = int(pre_filters * np.sqrt(2))
    total_filters.append(pre_filters)
    encoder.append([maxpool(), conv1d(pre_filters), PadLayerOdd2Even()])
  return encoder

def decoder():
  decoder = []
  for i in range(1, 12)[::-1]:
    decoder.append([mu.Merge.Concat(axis=2),
                    conv1d(total_filters[i]),
                    deconv1d(total_filters[i-1]),
                    conv1d(total_filters[i-1]),
                    Crop2Match(i-1)])
  decoder.append([mu.Merge.Concat(axis=2),conv1d(total_filters[0])])
  return decoder

def bottom_layer():
  return [[maxpool(), conv1d(int(total_filters[-1] * np.sqrt(2))),
           deconv1d(total_filters[-1]), conv1d(total_filters[-1]),
           Crop2Match(-1)]]

def edges():
  encode_str = '1'
  decode_str = ''
  str_temp = '1'
  for i in range(12):
    str_temp = '0' + str_temp
    encode_str = encode_str + ';' + str_temp
  for i in range(1, 13)[::-1]:
    str_temp = '0' * i + '1' + '0' * (12 - i) * 2 + '1'
    decode_str = decode_str + ';' + str_temp
  return encode_str+decode_str

def usleep(name):
  return mu.ForkMergeDAG(vertices=encoder(FIRST_FILTER)+bottom_layer()+decoder(),
                         edges=edges(), name=name)

def get_model():
  model = init_model()
  model.add(usleep('usleep'))
  return output_and_build(model)

