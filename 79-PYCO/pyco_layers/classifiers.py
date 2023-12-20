from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import tf
from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.hyper.conv import ConvBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from tframe import context
from tframe.operators.apis.attention import AttentionBase
# from caps_layers.models import *
class Transformer_Encoder(Layer, AttentionBase):

  full_name = 'encoder_layer'
  abbreviation = 'encoder'

  # @init_with_graph
  def __init__(self, d_model, d_ff, num_blocks, num_heads, maxlen,
               fc_dropout_rate, attention_dropout_rate, smooth=1, **kwargs):
    AttentionBase.__init__(self, **kwargs)
    self.num_heads = kwargs.get('num_heads', )
    self.d_model = d_model
    self.d_ff = d_ff
    self.num_blocks = num_blocks # 6
    self.num_heads =num_heads #8
    self.maxlen = maxlen
    self.fc_dropout_rate = fc_dropout_rate
    self.attention_dropout_rate = attention_dropout_rate
    self.smooth = smooth



  @single_input
  def _link(self, x, **kwargs):
    # x.shape [ ?, seq_le, n_dim,  n_channel] e.g. [bs, 64, 5, 128]
    input_shape = x.get_shape().as_list()

    # out [bs, 5, 64, 128]
    x = tf.transpose(x, [0, 2, 1, 3])

    # out [bs * 5, 64, 128]
    x = tf.reshape(x, [-1, input_shape[1], input_shape[-1]])


    # out [bs * 5, 64, 128]
    x += self.positional_encoding(x, self.maxlen)
    for i in range(self.num_blocks):
      x = self.multihead_attention(x, x, x, self.num_heads,
                                   self.attention_dropout_rate,
                                   )
      x = self.ff(x, [self.d_ff, self.d_model],
                  self.fc_dropout_rate,
                  )

    # out [bs, 5, 64, 128]
    x = tf.reshape(x, [-1, input_shape[2], input_shape[1], input_shape[3]])

    # out [bs, 64, 5, 128]
    x = tf.transpose(x, [0, 2, 1, 3])

    # out [bs * 64, 5, 128]
    x = tf.reshape(x, [-1, input_shape[-2], input_shape[-1]])

    return x


  def positional_encoding(self, inputs, maxlen, scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.
    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.get_shape().as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # position indices
      position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

      # First part of the PE function: sin and cos argument
      position_enc = np.array([
        [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
        for pos in range(maxlen)])

      # Second part, apply the cosine to even columns and sin to odds.
      position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
      position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
      position_enc = tf.convert_to_tensor(position_enc,
                                          tf.float32)  # (maxlen, E)

      # lookup
      outputs = tf.nn.embedding_lookup(position_enc, position_ind)

      return tf.to_float(outputs)

  def ff(self, inputs, num_units, dropout_rate=0., training=True,
         scope="positionwise_feedforward"):
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.
    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # Inner layer
      outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu,
                                )
      # outputs = self.dense(num_units[0], inputs, scope=scope+'_ff1')

      # Outer layer
      outputs = tf.layers.dense(outputs, num_units[1])
      # outputs = self.dense(num_units[1], outputs, scope=scope+'_ff2')
      # Residual connection
      outputs += inputs

      # Normalize
      outputs = self.ln(outputs)

      # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
      outputs = self.dropout(outputs, dropout_rate)

    return outputs

  def multihead_attention(self, queries, keys, values,
                          # key_masks,
                          num_heads=8,
                          attention_dropout_rate=0,
                          training=True,
                          # causality=False,
                          scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    d_model = queries.get_shape().as_list()[-1]
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # Linear projections
      Q = tf.layers.dense(queries, d_model, use_bias=True)  # (N, T_q, d_model)
      K = tf.layers.dense(keys, d_model, use_bias=True)  # (N, T_k, d_model)
      V = tf.layers.dense(values, d_model, use_bias=True)  # (N, T_k, d_model)
      # Split and concat
      Q_ = tf.concat(tf.split(Q, num_heads, axis=2),
                     axis=0)  # (h*N, T_q, d_model/h)
      K_ = tf.concat(tf.split(K, num_heads, axis=2),
                     axis=0)  # (h*N, T_k, d_model/h)
      V_ = tf.concat(tf.split(V, num_heads, axis=2),
                     axis=0)  # (h*N, T_k, d_model/h)
      # Attention
      outputs = self.scaled_dot_product_attention(Q_, K_, V_, attention_dropout_rate,
                                             training)

      # Restore shape
      outputs = tf.concat(tf.split(outputs, num_heads, axis=0),
                          axis=2)  # (N, T_q, d_model)

      # Residual connection
      outputs += queries

      # Normalize
      outputs = self.ln(outputs)

    return outputs

  def ln(self, inputs, epsilon=1e-8, scope="ln"):
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      inputs_shape = inputs.get_shape()
      params_shape = inputs_shape[-1:]

      mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
      beta = tf.get_variable("beta", params_shape,
                             initializer=tf.zeros_initializer())
      gamma = tf.get_variable("gamma", params_shape,
                              initializer=tf.ones_initializer())
      normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
      outputs = gamma * normalized + beta

    return outputs


  def scaled_dot_product_attention(self, Q, K, V, dropout_rate=0.,
                                   training=True,
                                   scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    key_masks: A 2d tensor with shape of [N, key_seqlen]
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      d_k = Q.get_shape().as_list()[-1]

      # dot product
      outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

      # scale
      outputs /= d_k ** 0.5

      # softmax
      outputs = tf.nn.softmax(outputs)
      attention = tf.transpose(outputs, [0, 2, 1])

      a = attention[:]
      tf.summary.image("attention", tf.expand_dims(attention[:1], -1))
      from tframe import context
      context.depot['self-attention'] = a

      # attention dropout
      # outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)
      outputs = self.dropout(outputs, dropout_rate)

      # weighted sum (context vectors)
      outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

class Attention(Layer):
  full_name = 'attn'
  abbreviation = 'attn'
  def __init__(self, attention_size, time_major=False, **kwargs):
    self.attention_size = attention_size
    self.time_major = time_major

  def _link(self, inputs, **kwargs):
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if self.time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    inputs_shape = inputs.shape
    sequence_length = inputs_shape[1].value  # the length of sequences processed in the antecedent RNN layer
    hidden_size = inputs_shape[2].value  # hidden size of the RNN layer

    # Attention mechanism
    W_omega = tf.Variable(tf.random_normal([hidden_size, self.attention_size],
                                           stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([self.attention_size],
                                           stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([self.attention_size],
                                           stddev=0.1))

    v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, hidden_size]),
                          W_omega) + tf.reshape(b_omega, [1, -1]))
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of Bi-RNN is reduced with attention vector
    output = tf.reduce_sum(inputs * tf.reshape(alphas,
                                               [-1, sequence_length, 1]),
                           1)
    context.depot['eason'] = alphas
    return output

