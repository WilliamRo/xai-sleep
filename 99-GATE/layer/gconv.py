from tframe import tf
from tframe.layers.hyper.conv import Conv1D as HyperConv1D
from tframe import hub as th

import typing as tp



class GatedConv1D(HyperConv1D):

    full_name = 'gconv1d'
    abbreviation = 'gconv1d'

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 dilations=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 expand_last_dim=False,
                 use_batchnorm=False,
                 filter_generator=None,
                 name: tp.Optional[str] = None,
                 **kwargs):

        # Call parent's constructor
        super(GatedConv1D, self).__init__(
            filters, kernel_size, strides, padding, dilations, activation,
            use_bias, kernel_initializer, bias_initializer, expand_last_dim,
            use_batchnorm, filter_generator, name, **kwargs)


    def get_layer_string(self, scale, full_name=False, suffix=''):
        activation = self._activation_string
        if self.dilations not in (None, 1): suffix += f'(di{self.dilations})'
        if callable(self.filter_generator): suffix += '[H]'
        if self.use_batchnorm: suffix += '->bn'
        if isinstance(activation, str): suffix += '->{}'.format(activation)
        result = super().get_layer_string(scale, full_name, suffix)
        return result


    def forward(self, x: tf.Tensor, filter=None, **kwargs):
        from tframe import mu
        def conv1d(x, filters, kernel_size, strides=1):
            """Conv1D layer"""
            return self.conv1d(x, filters, kernel_size, 'Gate'+str(strides),
                               strides=strides,
                               **kwargs)

        # Calculate gates
        c = self.channels
        s = self.kernel_size[0]
        gates = conv1d(x, c, s*10, 6)
        gates = conv1d(gates, c*2, s*5, 5)
        gates = conv1d(gates, c, s*2, 10)
        gates = conv1d(gates, x.shape[2].value, 5, 2)
        gates = mu.GlobalAveragePooling1D()(gates)
        gates = tf.reshape(gates, [-1, 1, x.shape[2].value])
        gates = tf.nn.sigmoid(gates)

        # Apply gates and return
        y = x * gates

        return y
