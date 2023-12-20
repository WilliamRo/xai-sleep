from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input


class ChannelGate(Layer, NeuroBase):

  full_name = 'channel_gate'
  abbreviation = 'channel_gate'
  def __init__(self, gate_channels, pool_types='avg', scope='gate'):
    self.gate_channels = gate_channels
    self.reduction = 16
    self.pool_types = pool_types
    self.scope = scope

  @single_input
  def _link(self, x, **kwargs):
    # x.shape (bs, L, C)
    # shape = x.shape.as_list()
    NeuroBase.__init__(self, **kwargs)
    self._data_format = 'channels_last'

    assert isinstance(x, tf.Tensor)
    shape = x.shape.as_list()
    assert len(shape) == 3
    y = tf.layers.average_pooling1d(
      x, pool_size=shape[1], strides=shape[1], data_format=self._data_format)

    y = tf.squeeze(y, 1)
    y = self.dense(self.gate_channels//self.reduction, y,
                   scope=f'{self.scope}_1', activation='relu')
    y = self.dense(self.gate_channels, y, scope=f'{self.scope}_2',
                   activation='sigmoid')

    return tf.multiply(x, y)

