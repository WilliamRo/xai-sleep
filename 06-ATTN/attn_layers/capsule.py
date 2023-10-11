# Capsule Network
import tensorflow as tf
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.hyper.conv import ConvBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
def squash(inputs, axis=-1):
  """
  The non-linear activation used in Capsule.
  It drives the length of a large vector to near 1 and small vector to 0
  :params inputs: vector to be squashed
  :params axis: the axis to squash
  :return: a Tensor with the same size as inputs
  """
  norm = tf.norm(inputs,axis=axis,keepdims=True)
  scale = norm**2/(1+norm**2)/(norm + 1e-8)

  return scale * inputs

class Capsule_layer(Layer, NeuroBase):
  full_name = 'PrimaryCapsule'
  abbreviation = 'PreCaps'

  def __init__(self, num_outputs, vec_len, layer_type='Conv'):
    self.num_outputs = num_outputs
    self.vec_len = vec_len
    self.layer_type = layer_type
    self.kernel_size = 9

  @single_input
  def _link(self, x, **kwargs):
    #
    # x.shape (bs, L, C) e.g.(bs, 45, 30)
    # shape = x.shape.as_list()
    NeuroBase.__init__(self, **kwargs)
    # return self.conv1d(
    #   x, self.channels, self.kernel_size, 'HyperConv1D', strides=self.strides,
    #   padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)
    capsules = self.conv1d(x, output_channels=self.num_outputs*self.vec_len,
                           scope='HyperConv1D',
                           self.kernel_size,strides=1)

    # self._data_format = 'channels_last'
    # # self.dense()
    # assert isinstance(x, tf.Tensor)
    # shape = x.shape.as_list()
    # assert len(shape) == 3
    # y = tf.layers.average_pooling1d(
    #   x, pool_size=shape[1], strides=1, data_format=self._data_format)
    # y = tf.reshape(y, shape=[-1, y.shape.as_list()[-1]])
    # y = self.dense(1, y, scope='se_f1')
    # y = self.dense(30, y, scope='se_f2')
    # y = tf.reshape(y, shape=(-1, 1, shape[2]))
    # y = tf.multiply(x, y)
    # # y = tf.broadcast_to(y, x.shape)
    return tf.multiply(x, y)
















if __name__ == '__main__':
  # Create a test tensor
  test_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

  # Apply the squash function
  squashed_tensor = squash(test_tensor)

  # Print the original and squashed tensors
  print("Original Tensor:")
  print(test_tensor.numpy())
  print("\nSquashed Tensor:")
  print(squashed_tensor.numpy())
