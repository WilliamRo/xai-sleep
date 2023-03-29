from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

from tframe import checker, linker
from tframe import hub, context, console
from tframe import initializers
from tframe import context
from tframe.layers.layer import LayerWithNeurons, Layer, single_input

class Dense(LayerWithNeurons):

  full_name = 'dense'
  abbreviation = 'dense'

  def __init__(
      self,
      num_neurons,
      activation=None,
      use_bias=True,
      weight_initializer='xavier_normal',
      bias_initializer='zeros',
      prune_frac=0,
      **kwargs):
    # Call parent's constructor
    LayerWithNeurons.__init__(
      self, activation, weight_initializer, use_bias, bias_initializer,
      prune_frac=prune_frac, **kwargs)

    self.num_neurons = checker.check_positive_integer(num_neurons)
    self.neuron_scale = [num_neurons]

  @property
  def structure_tail(self):
    activation = ''
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    return '({})'.format(self.num_neurons) + activation

  def forward(self, x, **kwargs):
    output = self.dense(self.num_neurons, x, activation=self._activation,
                      scope='dense')
    return output