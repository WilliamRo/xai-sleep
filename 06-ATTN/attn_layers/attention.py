from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tframe import checker
from tframe import hub as th
from tframe.operators.apis.neurobase import RNeuroBase
from tframe.operators.apis.attention import AttentionBase
from tframe import mu

from tframe import tf

from tframe.layers.layer import Layer
from tframe.layers.layer import single_input

from tframe.utils import get_scale
from tframe.core.decorators import init_with_graph
from tframe.core.function import Function

class EncoderLayer(Layer, AttentionBase):

  full_name = 'encoder_layer'
  abbreviation = 'encoder'

  # @init_with_graph
  def __init__(self, **kwargs):
    AttentionBase.__init__(self, **kwargs)
    self.num_heads = kwargs.get('num_heads', )

  def _split_heads(self, x, length, num_heads, depth):
    x = tf.reshape(x, (-1, num_heads, length // num_heads, depth))
    x = tf.transpose(x, perm=[0, 3, 2, 1])
    return x

  def _mha(self, q, k, v, num_heads=1, QK_dim=None, V_dim=None,
           output_dim=None, mask=None):
    """Multi-head attention.
       Must be called within a separate scope when multiple _mhas are called
       inside the same master scope."""
    # Check q, k, v and set default QK_dim if not provided
    min_qk_dim, q_len, k_len, v_len = self._check_qkv(q, k, v)
    if QK_dim is None: QK_dim = min_qk_dim

    # Calculate Q, K, V, where Q, K must be transformed by q and k
    Q = q
    K = self.conv1d(k, QK_dim, filter_size=7, scope='kconv1')
    V = self.conv1d(v, V_dim, filter_size=7, scope='vconv1')

    # V is allowed to be v
    # TODO: if V_dim is not specified, simply stack v may not be appropriate.
    #       since the output attention may be grown large

    # Split head if necessary
    if num_heads > 1:
      Q = self._split_heads(Q, q_len, num_heads, QK_dim)
      K = self._split_heads(K, k_len, num_heads, QK_dim)
      V = self._split_heads(V, v_len, num_heads, V_dim)


    # Apply attention, out shape = (bs[, num_heads], q_len, [Vv]_dim)
    attention, p_attn = self.scaled_dot_prod_attention(Q, K, V, mask, return_a=True)

    # Reshape back if necessary
    if num_heads > 1:
      attention = tf.transpose(attention, perm=[0, 3, 2, 1]) # type: tf.Tensor
      last_dim = attention.shape.as_list()[-1]
      attention = tf.reshape(attention, (-1, q_len, last_dim))
    attention = self.dropout(attention, 0.1)

    # Calculate output and return
    if output_dim is not None:
      output = tf.transpose(attention, perm=[0, 2, 1])
      output = self.dense(output_dim, output, scope='output')
      output = tf.transpose(output, perm=[0, 2, 1])



    return output


  @single_input
  def _link(self, x, **kwargs):
    # x.shape = [?, L, C], e.g., [?, 78, 30] ->[?, 80, 30]
    d_model = 100
    q = self.conv1d(x, 30, 7, 'qconv')
    y = self._mha(q, x, x, num_heads=self.num_heads, V_dim=30, output_dim=d_model)
    y = self.dropout(y, 0.1)
    y = q + y
    return y

