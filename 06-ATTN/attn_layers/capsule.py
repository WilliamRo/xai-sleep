# Capsule Network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import tf

from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.hyper.conv import ConvBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input


def squash(inputs, axis=-2):
  """
  The non-linear activation used in Capsule.
  It drives the length of a large vector to near 1 and small vector to 0
  :params inputs: vector to be squashed
  :params axis: the axis to squash
  :return: a Tensor with the same size as inputs
  """
  norm = tf.norm(inputs, axis=axis, keepdims=True)
  scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)

  return scale * inputs


def routing(input, b_ij, num_outputs=5, num_dims=16, routing_times=3):
  """
	routing algarithm
	"""
  # input_size before caps (bs, caps_nums, 1, vec_len, 1) e.g. (bs, 1152, 1, 8, 1)
  input_shape = input.shape.as_list()

  W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                      dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
  biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

  # calc u_hat
  input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
  # assert input.get_shape() == [bs, 1152, 80, 8, 1]
  u_hat = W * input
  u_hat = tf.reduce_sum(u_hat, 3, True)
  u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
  # assert u_hat.get_shape() == [bs, 1152, 10, 16, 1]

  u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

  for r_iter in range(routing_times):
    # [bs, 1152, 5, 1, 1]
    # b_ij shape [1152, 5, 1, 1]
    c = tf.nn.softmax(b_ij, axis=1)
    if r_iter == routing_times - 1:
      s_J = tf.multiply(s_J, u_hat)
      s_J = tf.reduce_sum(s_J, 1, True) + biases
      v_J = squash(s_J)

      # assert v_J.get_shape()== [bs, 1, num_outputs, num_dims, 1]

      pass
      # at the last time, use 'u_hat'
    else:
      s_J = tf.multiply(c, u_hat_stopped)
      s_J = tf.reduce_sum(s_J, 1, True) + biases
      v_J = squash(s_J)

      v_J_tited = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
      u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tited, axis=3, keepdims=True)

      b_ij += u_produce_v

  return v_J


class PrimaryCaps(Layer, NeuroBase):
  full_name = 'PrimaryCapsule'
  abbreviation = 'PriCaps'

  def __init__(self, num_outputs, vec_len, layer_type='Conv', **kwargs):
    self.num_outputs = num_outputs
    self.vec_len = vec_len
    self.layer_type = layer_type
    self.kernel_size = 9
    self.batch_size = kwargs.get('batch_size', None)

  @single_input
  def _link(self, x, **kwargs):
    # x.shape (bs, L, C) e.g.(bs, 45, 30)
    NeuroBase.__init__(self, **kwargs)

    # output (bs, L`, C`) e.g. (bs, 113, 256)
    capsules = self.conv1d(x, self.num_outputs * self.vec_len, self.kernel_size,
                           'HyperConv1D',
                           strides=2)
    capsules_shape = capsules.shape.as_list()
    capsules = tf.reshape(capsules, (-1, capsules_shape[1] * capsules_shape[2] // self.vec_len, self.vec_len))
    if self.batch_size is None: self.batch_size = capsules_shape[0]
    # output (bs, L`, self.vec_len) e.g. (bs, 113*32, 8 )
    # capsules = tf.reshape(capsules, (self.batch_size, -1, self.vec_len))

    return squash(capsules)


class DigitsCaps(Layer, NeuroBase):
  full_name = 'DigitsCapsule'
  abbreviation = 'DitCaps'

  def __init__(self, num_outputs, vec_len, layer_type='Conv', **kwargs):
    self.num_outputs = num_outputs
    self.vec_len = vec_len
    self.layer_type = layer_type
    # self.kernel_size = 9
    self.batch_size = kwargs.get('batch_size', None)

  @single_input
  def _link(self, x, **kwargs):
    # x.shape (bs, caps_nums, vec_len) e.g.(bs, 1152, 8)
    NeuroBase.__init__(self, **kwargs)

    # input_size before caps (bs, caps_nums, 1, vec_len, 1) e.g. (bs, 1152, 1, 8, 1)
    input_shape = x.shape.as_list()
    if self.batch_size is None: self.batch_size = input_shape[0]
    capsules = tf.reshape(x, (-1, input_shape[1], 1, input_shape[-1], 1))
    # capsules = tf.reshape(x, (self.batch_size, -1, 1, input_shape[-1], 1))

    # routing
    b_ij = tf.constant(np.zeros([input_shape[1], self.num_outputs, 1, 1], dtype=np.float32))

    output = routing(capsules, b_ij, self.num_outputs, self.vec_len)
    capsules = tf.squeeze(output, axis=1)

    return squash(capsules)

class Caps_finalize(Layer):
  full_name = 'Caps_finalize'
  abbreviation = 'capsfinal'
  def __init__(self, **kwargs):
    pass

  def _link(self, inputs, **kwargs):
    # input [bs, num_outputs, vec_len, 1] => [bs, num_outputs, 1, 1]

    self.v_length = tf.sqrt(tf.reduce_sum(tf.square(inputs), 2, True) +1e-9)
    self.softmax_v = tf.nn.softmax(self.v_length, axis=1)
    # self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
    # self.argmax_idx = tf.squeeze(self.argmax_idx, axis=-1)

    # => [bs, num_outputs]
    softmax_v = tf.squeeze(self.softmax_v, axis=[-1, -2])
    return softmax_v



if __name__ == '__main__':
  final = Caps_finalize()
  final(1)

