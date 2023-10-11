from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe import tf

import tframe as tfr
from tframe.utils.arg_parser import Parser
from tframe.layers.layer import Layer
from tframe.layers.layer import single_input
from tframe.layers.merge import Merge

from tframe.utils.misc import get_scale

from tframe import activations
from tframe import hub
from tframe import initializers
from tframe import regularizers
from tframe import pedia


class Pad_Merge(Merge):

  Pad_CONCAT = 'pad_concat'
  PROD = pedia.prod
  SUM = pedia.sum
  CONCAT = pedia.concat
  CROSS_CONCAT = 'cross-concat'
  CONCAT_SUM = 'concat-sum'
  HIGHWAY = 'highway'

  def __init__(self, merge_method, **kwargs):
    # super(self,Pad_Merge).__init__(merge_method)
    self.full_name, self.abbreviation = merge_method, merge_method
    self.merge_method = merge_method
    self._axis = kwargs.get('axis', -1)
    self.pad = kwargs.get('pad', 0) *2 + 1

  def _link(self, *input_list, **kwargs):
    # Check input_list
    assert len(input_list) > 0
    if len(input_list) == 1: input_list = input_list[0]
    if not (isinstance(input_list, (list, tuple)) and len(input_list) > 1):
      raise ValueError('!! Illegal input tensors flow into merge layer.')

    # Slice if necessary
    input_list = self._check_input_list(input_list)

    # Merge according to specification

    if self.merge_method == 'pad_concat' and self.pad:
      input1 = input_list[0]
      input2 = input_list[1]
      input_shape = input1.get_shape().as_list()
      input_shape2 = input2.get_shape().as_list()

      input1 = tf.reshape(input1, (-1,
                                   self.pad,
                                   input_shape[1]//self.pad,
                                   input_shape[-1]))
      input2 = tf.reshape(input2,
                          (-1,
                           self.pad,
                           input_shape2[1]//self.pad,
                           input_shape2[-1]))

      output =  tf.concat([input1, input2], axis=self._axis+1)
      output_shape = output.get_shape().as_list()
      return tf.reshape(output, (-1,output_shape[1]*output_shape[2],input_shape[-1]))

    else: raise KeyError('!! Unknown merge method {}'.format(self.merge_method))


