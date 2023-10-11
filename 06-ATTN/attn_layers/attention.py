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

class SelfAttention(Layer, AttentionBase):

  full_name = 'self_attention'
  abbreviation = 'self_attn'

  # @init_with_graph
  def __init__(self, **kwargs):
    AttentionBase.__init__(self, **kwargs)



  def _mha(self, q, k, v, num_heads=1, QK_dim=None, V_dim=None,
           output_dim=None, mask=None):
    """Multi-head attention.
       Must be called within a separate scope when multiple _mhas are called
       inside the same master scope."""
    # Check q, k, v and set default QK_dim if not provided
    min_qk_dim, q_len, k_len, v_len = self._check_qkv(q, k, v)
    if QK_dim is None: QK_dim = min_qk_dim

    # Calculate Q, K, V, where Q, K must be transformed by q and k
    Q = self.dense(num_heads * QK_dim, q, scope='query')
    K = self.dense(num_heads * QK_dim, k, scope='key')
    # V is allowed to be v
    if V_dim is not None: V = self.dense(num_heads * V_dim, v, scope='value')
    elif num_heads == 1: V = v
    else: V = tf.stack([v] * num_heads, axis=-3)
    # TODO: if V_dim is not specified, simply stack v may not be appropriate.
    #       since the output attention may be grown large

    # Split head if necessary
    if num_heads > 1:
      Q = self._split_heads(Q, q_len, num_heads, QK_dim)
      K = self._split_heads(K, k_len, num_heads, QK_dim)
      if V_dim is not None: V = self._split_heads(V, v_len, num_heads, V_dim)

    # Apply attention, out shape = (bs[, num_heads], q_len, [Vv]_dim)
    attention = self.scaled_dot_prod_attention(Q, K, V, mask)

    # Reshape back if necessary
    if num_heads > 1:
      attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # type: tf.Tensor
      last_dim = num_heads * attention.shape.as_list()[-1]
      attention = tf.reshape(attention, (-1, q_len, last_dim))

    # Calculate output and return
    output = (attention if output_dim is None
              else self.dense(output_dim, attention, scope='output'))
    return output


  @single_input
  def _link(self, x, **kwargs):
    # x.shape = [?, L, C], e.g., [?, 45, 30]
    afr_reduce_cnn_size = 30
    y = self._mha(x, x, x, num_heads=5, output_dim=afr_reduce_cnn_size)
    # output = tf.layers.average_pooling1d(
    #   input_, pool_size=shape[1], strides=1, data_format=self._data_format)
    # output = tf.reshape(output, shape=[-1, output.shape.as_list()[-1]])
    return y

