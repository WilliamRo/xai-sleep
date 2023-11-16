from fre_ops.fre_filter import DaFilter
from fre_ops.stft_layer import STFT

from tframe import mu
from tframe.layers.pooling import ReduceMean



def get_initial_model():
  from fre_core import th

  model = mu.Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))

  return model



def finalize(model: mu.Classifier, flatten=False, use_gap=False):
  from fre_core import th

  if use_gap:
    model.add(ReduceMean(axis=1))
    # model.add(mu.GlobalAveragePooling1D())
    # model.add(mu.Flatten())
    model.add(mu.Activation('softmax'))
  else:
    if flatten: model.add(mu.Flatten())

    model.add(mu.Dense(num_neurons=th.output_dim))
    model.add(mu.Activation('softmax'))

  model.build(metric=['f1', 'accuracy'], batch_metric='accuracy')

  return model



def add_deep_sleep_net_lite(model: mu.Classifier, N: int):
  from fre_core import th

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



