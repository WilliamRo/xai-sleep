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


class BatchReshape(Layer):
  full_name = 'batch_reshape'
  abbreviation = 'b_rs'
  def __init__(self, reverse=False, squeeze=False, **kwargs):
    self.reverse = reverse
    self.split_num = kwargs.get('split_num', None)
    # self.squeeze = kwargs.get('squeeze', False)
    self.squeeze = squeeze
  def _link(self, x, **kwargs):
    from tframe import hub as th
    if self.split_num is None: self.split_num = th.epoch_num
    # input shape [bs, L * epoch_num, C]
    if self.squeeze: x = tf.squeeze(x, -1)
    input_shape = x.get_shape().as_list()

    if not self.reverse:
      # [bs, L * epoch_num, C] -> [bs*epoch_num, L, C]
      x = tf.reshape(x, [-1, self.split_num, input_shape[1]//self.split_num, input_shape[-1]])
      x = tf.reshape(x, [-1, input_shape[1]//self.split_num, input_shape[-1]])

    else:
      # [bs*epoch_num, d_feat] -> [bs, epoch_num, d_feat]
      if len(input_shape) == 2:
        x = tf.reshape(x, [-1, self.split_num, input_shape[-1]])
      elif len(input_shape) == 3:
        x = tf.reshape(x, [-1, self.split_num, input_shape[1], input_shape[2]])
    # print(x.get_shape())
    # x.shape [bs * epoch_num, L, C]
    return x



class STFT(Layer):
  full_name = 'STFT'
  abbreviation = 'stft'

  def __init__(self, frame_length, frame_step, fft_length, **kwargs):

    self.frame_length = frame_length
    self.frame_step = frame_step
    self.fft_length = fft_length
    self.hamming_window = tf.signal.hamming_window(self.frame_length)
    self.discard = kwargs.get('discard', False)


  def _link(self, x, **kwargs):

    stft_list = []
    n_channel = x.get_shape().as_list()[-1]
    for i in range(n_channel):
      stft = tf.signal.stft(x[:, :, i], frame_length=self.frame_length,
                          frame_step=self.frame_step, fft_length=self.fft_length,
                          pad_end=False)
      if self.discard: stft = stft[:, :, 1:]
      stft_amplitude = tf.math.abs(stft)
      stft_list.append(stft_amplitude)

    # stft_concatenated = tf.concat(stft_list, axis=-1)
    stft_concatenated = tf.stack(stft_list, -1)
    print(stft_concatenated)
    return stft_concatenated
