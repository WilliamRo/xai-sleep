from tframe.layers.hyper.hyper_base import HyperBase
import tensorflow as tf


class BiLSTM(HyperBase):
  full_name = 'dense'
  abbreviation = 'dense'

  def __init__(
    self,
    # input_size,
    batch_size=64,
    hidden_size=512,
    num_layer=2,
    use_dropout=False,
    return_last=True,
    **kwargs):

    self.hidden_size = hidden_size
    self.num_layer = num_layer
    self.use_dropout = use_dropout
    self.batch_size = batch_size
    self.return_last = return_last
    self._activation = None
    self.num_neurons = 0
    self.fw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell \
      ([self.lstm_cell() for _ in range(self.num_layer)], state_is_tuple=True)
    self.bw_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell \
      ([self.lstm_cell() for _ in range(self.num_layer)], state_is_tuple=True)
    self.fw_initial_state = self.fw_cell.zero_state(self.batch_size, tf.float32)
    self.bw_initial_state = self.bw_cell.zero_state(self.batch_size, tf.float32)

  def lstm_cell(self):
    cell = tf.compat.v1.nn.rnn_cell.LSTMCell(self.hidden_size,
                                             use_peepholes=True,
                                             state_is_tuple=True,
                                             reuse=tf.compat.v1.get_variable_scope().reuse)
    # if self.use_dropout_sequence:
    #   keep_prob = 0.5 if self.is_train else 1.0
    #   cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
    #     cell,
    #     output_keep_prob=keep_prob
    #   )
    return cell

  @property
  def structure_tail(self):
    activation = 'lstm'
    if self._activation is not None:
      activation = '->act'
      if isinstance(self._activation_string, str):
        activation = '->' + self._activation_string
    return '({})'.format(self.num_neurons) + activation
    # return activation

  def forward(self, x, **kwargs):
    # Initial state of RNN
    x = tf.expand_dims(x, axis=2)

    # Feedforward to MultiRNNCell
    list_rnn_inputs = tf.unstack(x, axis=1)[0:5]
    # outputs, fw_state, bw_state = tf.nn.bidirectional_rnn(
    outputs, fw_state, bw_state = tf.compat.v1.nn.static_bidirectional_rnn(
      cell_fw=self.fw_cell,
      cell_bw=self.bw_cell,
      inputs=list_rnn_inputs,
      initial_state_fw=self.fw_initial_state,
      initial_state_bw=self.bw_initial_state
    )

    if self.return_last:
      outputs = outputs[-1]
    else:
      outputs = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.hidden_size*2])

    return outputs
