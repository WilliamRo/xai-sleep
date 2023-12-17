from tframe import mu
from pyco_layers.eason_classifier import EasonClassifier
from tframe.layers.pooling import ReduceMean



def get_initial_model():
  from pyco_core import th

  model = EasonClassifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))

  return model



def finalize(model: mu.Classifier, flatten=False, use_gap=False):
  from pyco_core import th

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

def decoder(model, flatten=False, use_gap=False):
  from pyco_core import th


  model.add(mu.Flatten())
  model.add(mu.Dense(num_neurons=th.output_dim))
  model.add(mu.Activation('softmax'))
  model.add(mu.Dense(1024))
  model.add(mu.Dense(3000))

  model.build(metric=['f1', 'accuracy'], batch_metric='accuracy', loss='mse')

  return model


def add_deep_sleep_net_lite(model: mu.Classifier, N: int):
  from pyco_core import th

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



def add_densely_connected_temporal_pyramids(model: mu.Classifier):
  from pyco_core import th

  # Encoder args
  en_filter, en_ks = th.dtp_en_filters, th.dtp_en_ks
  en_strides = en_ks // 2

  # Add encoder
  model.add(mu.HyperConv1D(en_filter, en_ks, strides=en_strides))

  # Construct TPs using fmDAG
  M, R, DC, DKS = th.dtpM, th.dtpR, th.filters, th.kernel_size
  BC = DC // 2
  bottle_neck = lambda c=BC: mu.HyperConv1D(
    c, 1, use_batchnorm=th.use_batchnorm, activation=th.activation)
  concat = lambda: mu.Merge.Concat()

  vertices, edges = [], '1'
  for r in range(R):
    for m in range(M):
      index = r * M + m

      tpb = []
      if index > 1: tpb.extend([concat(), bottle_neck()])
      tpb.extend([mu.HyperConv1D(DC, DKS, dilations=2**m,
                                 activation=th.activation),
                  bottle_neck()])
      vertices.append(tpb)
      edges += ';0' + '1' * (index + 1)

  # Add final merge layer
  vertices.append([concat(), bottle_neck(5)])

  model.add(mu.ForkMergeDAG(vertices, edges, name='DTP'))













