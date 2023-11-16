from tframe import tf
from tframe.layers.layer import Layer, single_input



class DaFilter(Layer):
  """"""
  abbreviation = 'da_filter'
  full_name = abbreviation

  def __init__(self, ks=32):
    self.ks = ks

  @property
  def structure_tail(self): return f'({self.ks})'

  @single_input
  def _link(self, x, **kwargs):
    """Remove low-fre in each channel using depth-wise conv1D"""
    # x.shape = [?, L, C] (NWC format), e.g., [?, fs*30, 2]
    C = x.shape.as_list()[-1]
    xs = tf.split(x, num_or_size_splits=C, axis=-1)

    filters = tf.constant(
      [1 / self.ks] * self.ks, dtype=tf.float32, shape=[self.ks, 1, 1])

    xs_low_fre = [tf.nn.conv1d(x, filters=filters, padding='SAME') for x in xs]
    x_low_fre = tf.concat(xs_low_fre, axis=-1)

    return x - x_low_fre
