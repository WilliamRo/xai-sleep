from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

class SE_layer(Layer, NeuroBase):

  full_name = 'se_layer'
  abbreviation = 'se'


  @single_input
  def _link(self, x, **kwargs):
    # x.shape (bs, 45, 30)
    # shape = x.shape.as_list()
    NeuroBase.__init__(self, **kwargs)
    self._data_format = 'channels_last'

    assert isinstance(x, tf.Tensor)
    shape = x.shape.as_list()
    assert len(shape) == 3
    y = tf.layers.average_pooling1d(
      x, pool_size=shape[1], strides=1, data_format=self._data_format)
    y = tf.reshape(y, shape=[-1, y.shape.as_list()[-1]])
    y = self.dense(1, y, scope='se_f1', activation='relu')
    y = self.dense(30, y, scope='se_f2', activation='sigmoid')
    y = tf.reshape(y, shape=(-1, 1, shape[2]))

    return tf.multiply(x, y)

