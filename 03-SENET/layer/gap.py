from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tframe.layers.layer import Layer
from tframe.core.decorators import init_with_graph

import tensorflow as tf
class GlobalAveragePooling1D(Layer):

    full_name = 'gate1_avg'
    abbreviation = 'gap1d'

    @init_with_graph
    def __init__(self, data_format='channels_last', flatten=True, **kwargs):
        self._data_format = data_format
        assert data_format == 'channels_last'
        self._flatten = flatten
        self._kwargs = kwargs

    def _link(self, input_, **kwargs):
        assert isinstance(input_, tf.Tensor)
        shape = input_.shape.as_list()
        assert len(shape) == 3
        # dimension = kwargs.get('dimension', 'channel')
        dimension = kwargs['dimension']
        if dimension == 'channel':
            output = tf.reduce_mean(input_, axis=1)
        if dimension == 'spatial':
            output = tf.reduce_mean(input_, axis=2)
        return output
