from tframe import tf
from tframe import context

from tframe.layers.hyper.conv import ConvBase



class GatedConv1D(ConvBase):
  """Perform 1D convolution on a channel-last data"""

  full_name = 'gconv1d'
  abbreviation = 'gconv1d'

  class Configs(ConvBase.Configs):
    kernel_dim = 1

  def forward(self, x: tf.Tensor, filter=None, **kwargs):
    from tframe import mu

    # x.shape = [?, L, C]
    # gate.shape = [?, 1, C] \in (0, 1)
    gate = self.conv1d(
      x, self.channels, self.kernel_size, 'Gate', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)
    gate = tf.reduce_mean(gate, axis=1, keepdims=True)
    gate = tf.nn.sigmoid(gate)

    # (1)
    context.add_tensor_to_export(f'{self.full_name}', gate)

    #
    y = self.conv1d(
      x, self.channels, self.kernel_size, 'GatedConv1D', strides=self.strides,
      padding=self.padding, dilations=self.dilations, filter=filter, **kwargs)
    return gate * y
