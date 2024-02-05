import numpy as np

from tframe import tf
from tframe.layers.layer import Layer, single_input



class STFT(Layer):
  """"""
  abbreviation = 'stft'
  full_name = abbreviation

  def __init__(self, max_fre=30):
    self.max_fre = max_fre

  @property
  def structure_tail(self): return f'({self.max_fre})'

  @single_input
  def _link(self, x, **kwargs):
    # x.shape = [?, L, C] (NWC format), e.g., [?, fs*30, 2]
    fs = 128
    nperseg = 2 * fs
    # Transpose signal to [..., samples]
    x = tf.transpose(x, [0, 2, 1])
    stft = tf.signal.stft(x, frame_length=nperseg, frame_step=nperseg // 2)
    y = tf.abs(stft)
    # y.shape = [..., frames, bins], bins = nperseg // 2 + 1

    # Cut y
    F = int(nperseg // 2 / (fs / 2) * self.max_fre)
    y = y[..., :F]

    # Transpose signal back
    y = tf.transpose(y, [0, 2, 3, 1])
    return y



class FrequencyEstimator(Layer):
  abbreviation = 'frest'
  full_name = abbreviation
  def __init__(self, max_fre=20):
    self.max_fre = max_fre

  @property
  def structure_tail(self): return f'({self.max_fre})'

  @single_input
  def _link(self, x: tf.Tensor, **kwargs):
    # x.shape = [?, L, C] (NWC format), e.g., [?, fs*30, 2]
    fs = 128
    nperseg = 2 * fs
    # Transpose signal to [..., samples]
    x = tf.transpose(x, [0, 2, 1])
    stft = tf.signal.stft(x, frame_length=nperseg, frame_step=nperseg // 2)
    y = tf.abs(stft)
    # y.shape = [..., frames, bins], bins = nperseg // 2 + 1

    # Cut y
    f = np.linspace(0, fs / 2, num=nperseg // 2 + 1)
    max_F = int(nperseg // 2 / (fs / 2) * self.max_fre)
    f = f[2:max_F + 1]
    f = tf.constant(f, dtype=x.dtype, shape=[len(f)])
    y = y[..., 2:max_F + 1]

    frest = tf.reduce_sum(y * f, axis=-1) / tf.reduce_sum(y, axis=-1)
    # frest.shape = [?, C, T]

    frest_norm = frest / max_F

    # Transpose signal back
    return tf.transpose(frest_norm, [0, 2, 1])
