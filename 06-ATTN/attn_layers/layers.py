from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tframe import tf

from tframe.operators.apis.neurobase import NeuroBase
from tframe.layers.hyper.conv import ConvBase
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from attn_layers.attention import SelfAttention
from attn_layers.SE_layer import SE_layer
from attn_layers.merge import Pad_Merge

class Gelu(Layer):
  full_name = 'gelu'
  abbreviation = 'gelu'
  def __init__(self, **kwargs):
    pass

  def _link(self, x, **kwargs):

    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf

class STFT(Layer):
  full_name = 'STFT'
  abbreviation = 'stft'

  def __init__(self, sfreq=100):
    self.sfreq = sfreq

  def _link(self, x, **kwargs):
    from scipy.signal import stft
    f, t, Zxx = stft(x, fs=self.sfreq, nperseg=256)
    spectrum = np.abs((Zxx))
    return spectrum