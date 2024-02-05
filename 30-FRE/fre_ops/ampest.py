from tframe import tf
from tframe.layers.layer import Layer, single_input



class AmplitudeEstimator(Layer):
  """"""
  abbreviation = 'ampest'
  full_name = abbreviation

  def __init__(self, ks=128):
    self.ks = ks

  @property
  def structure_tail(self): return f'({self.ks})'

  @single_input
  def _link(self, x, **kwargs):
    """Remove low-fre in each channel using depth-wise conv1D"""
    # x.shape = [?, L, C] (NWC format), e.g., [?, fs*30, 2]
    ks, stride = self.ks, self.ks // 2
    x_max = tf.nn.max_pool1d(x, self.ks, stride, padding='VALID')
    x_min = -tf.nn.max_pool1d(-x, self.ks, stride, padding='VALID')
    amp = x_max - x_min

    return amp
