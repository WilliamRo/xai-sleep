from tframe import mu
from tframe import tf
from tframe import context
from tframe.models import Recurrent, Feedforward



def get_initial_model():
  from rnn_core import th

  model = mu.Classifier(mark=th.mark,
                        net_type=Recurrent if th.use_rnn else Feedforward)

  model.add(mu.Input(sample_shape=th.input_shape))

  return model



def finalize(model: mu.Classifier):
  from rnn_core import th

  model.add(mu.Dense(num_neurons=th.output_dim))
  model.add(mu.Activation('softmax'))

  model.build(metric=['f1', 'accuracy'], batch_metric='accuracy',
              loss=th.loss_string)

  return model



def add_deep_sleep_net_lite(model: mu.Classifier, N: int):
  from rnn_core import th

  conv = lambda ks, c, s=1: mu.HyperConv1D(
    filters=c, kernel_size=ks, strides=s,
    use_batchnorm=th.use_batchnorm, activation=th.activation)
  pool = lambda k: mu.MaxPool1D(pool_size=k, strides=k)
  dp = lambda: mu.Dropout(th.dropout)

  fs = 128
  vertices = [[conv(fs // 2, N, fs // 16), pool(8), dp(),
               conv(8, 2 * N), conv(8, 2 * N), conv(8, 2 * N), pool(4)],
              [conv(fs * 4, N, fs // 2), pool(4), dp(),
               conv(6, 2 * N), conv(6, 2 * N), conv(6, 2 * N), pool(2)],
              [mu.Merge(mu.Merge.CONCAT, axis=1), dp()]]
  fm = mu.ForkMergeDAG(vertices, edges='1;10;011')

  model.add(fm)
