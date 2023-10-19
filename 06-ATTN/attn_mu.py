from tframe import mu

from tframe.layers.pooling import ReduceMean
# from tframe.layers.pooling import GlobalAveragePooling1D as gap
# from attn_layers.pool import AveragePooling1D
# from attn_layers.pool import GlobalAveragePooling1D as gap
from attn_layers.attention import SelfAttention
from attn_layers.SE_layer import SE_layer
from attn_layers.merge import Pad_Merge
from attn_layers.layers import Gelu
from attn_layers.layers import  STFT


# region: Initial and Finalize Model

def get_initial_model():
  from attn_core import th

  model = mu.Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))

  return model


def finalize(model: mu.Classifier, flatten=False, use_gap=False):
  from attn_core import th
  if use_gap:
    model.add(ReduceMean(axis=1))
    # model.add(mu.GlobalAveragePooling1D())
    # model.add(mu.Flatten())
    model.add(mu.Activation('softmax'))
  else:
    if flatten: model.add(mu.Flatten())

    model.add(mu.Dense(num_neurons=th.output_dim))
    model.add(mu.Activation('softmax'))

  model.build(metric=['loss', 'f1', 'accuracy'], batch_metric='accuracy', loss=th.loss_string)

  return model

def Caps_finalize(model: mu.Classifier):
  from attn_layers.capsule import Caps_finalize
  final = Caps_finalize()
  model.add(final)
  model.build(metric=['f1', 'accuracy'], batch_metric='accuracy')
  return model
# endregion: Initial and Finalize Model

# region: MSCNN
def add_deep_sleep_net_lite(model: mu.Classifier, N: int):
  from attn_core import th
  # th.activation = 'Gelu'
  dprate = 0.5
  conv = lambda ks, c, s=1: mu.HyperConv1D(
    filters=c, kernel_size=ks, strides=s,
    use_batchnorm=th.use_batchnorm, activation=th.activation)
  pool = lambda k: mu.MaxPool1D(pool_size=k, strides=k)
  dp = lambda: mu.Dropout(dprate)
  fs = 128
  # vertices = [[conv(fs // 2, N, fs // 16), mu.MaxPool1D(8, 4), dp(),
  #              conv(8, 2 * N), conv(8, 2 * N), conv(8, 2 * N), pool(4)],
  #             [conv(fs * 4, N, fs // 2), mu.MaxPool1D(4, 2), dp(),
  #              conv(6, 2 * N), conv(6, 2 * N), conv(6, 2 * N), pool(2)],
  #             [mu.Merge(mu.Merge.CONCAT, axis=1), dp()]]
              # [Pad_Merge('pad_concat', pad=th.epoch_pad, axis=1), dp()]]

  vertices = [[mu.Conv1D(64, 50, 6, use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.MaxPool1D(8, 2), dp(),
               mu.Conv1D(128, 8, 1, use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.Conv1D(128, 8, 1,  use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.MaxPool1D(4, 4 )
               ],
              [mu.Conv1D(64, 400, 50,  use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.MaxPool1D(4, 2), dp(),
               mu.Conv1D(128, 6, 1,  use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.Conv1D(128, 6, 1,  use_batchnorm=th.use_batchnorm),
               Gelu(),
               mu.MaxPool1D(2, 2)
               ],
              # [mu.Merge(mu.Merge.CONCAT, axis=1), dp()]]
              [Pad_Merge('pad_concat', pad=th.epoch_pad, axis=1), dp()]]
  fm = mu.ForkMergeDAG(vertices, edges='1;10;011')
  model.add(fm)


def add_AFR(model: mu.Classifier):
  vertices = [[mu.Conv1D(30, 1, activation='Relu', use_batchnorm=True),
               mu.Conv1D(30, 1, use_batchnorm=True),
               SE_layer()
               ],
              [mu.Conv1D(30, 1, use_batchnorm=True)],
              [mu.Merge(mu.Merge.SUM)]]
  afr = mu.ForkMergeDAG(vertices, edges='1;10;011')
  model.add(afr)


# endregion: MSCNN

# region: MHA
def add_EncodeLayer(model):
  vertices = [[mu.LayerNormalization(),
               SelfAttention(num_heads=3),
               mu.Dropout()],
              [mu.Merge(mu.Merge.SUM)],
              [mu.LayerNormalization(), mu.Dense(120), mu.Activation('relu'),
               mu.Dense(30), mu.Dropout()],
              [mu.Merge(mu.Merge.SUM)]
              ]
  encoder = mu.ForkMergeDAG(vertices, edges='1;11;001;0011')
  model.add(encoder)

  model.add(mu.LayerNormalization())

# endregion: MHA


# region: Capsule

# endregionL Capsule

# region: STFT
def add_stft_layer(model, sfreq=100):
  stft = STFT(sfreq)
  model.add(stft)
  return model


# endregion: STFT