from layer.bilstm import BiLSTM
from tframe import Classifier
from tframe import mu
from dsn_core import th

# region: basic construction
def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model

def finalize(model):
  assert isinstance(model, Classifier)
  model.add(mu.Flatten())
  model.add(mu.Dense(th.output_dim, use_bias=False, prune_frac=0.5))
  model.add(mu.Activation('softmax'))
  # Build model
  model.build(batch_metric=['accuracy'])
  return model
# endregion

# region: module api
def conv1d(kernel_size, filters, strides=1):
  """Conv1D layer"""
  return mu.Conv1D(filters, kernel_size, strides,
                   use_batchnorm=th.use_batchnorm,
                   activation=th.activation)

def maxpool(pool_size, strides):
  """Maxpool layer"""
  return mu.MaxPool1D(pool_size, strides)

def dropout():
  """Dropout"""
  return mu.Dropout(0.5)

def flatten():
  return mu.Flatten()
# endregion

# region: functional net
def feature_extracting_net(name, n=32):
  return mu.ForkMergeDAG(vertices=[
    [conv1d(50, 2 * n, 6), maxpool(8, 8), dropout(), conv1d(8, 4 * n),
     conv1d(8, 4 * n), conv1d(8, 4 * n), maxpool(4, 4), flatten()],
    [conv1d(400, 2 * n, 50), maxpool(4, 4), dropout(), conv1d(6, 4 * n),
     conv1d(6, 4 * n), conv1d(6, 4 * n), maxpool(2, 2), flatten()],
    [mu.Merge.Concat(axis=1), dropout()]],
    edges='1;10;011', name=name)

def dsn_net(name, n=32):
  return mu.ForkMergeDAG(vertices=[
    [conv1d(50, 2 * n, 6), maxpool(8, 8), dropout(), conv1d(8, 4 * n),
     conv1d(8, 4 * n), conv1d(8, 4 * n), maxpool(4, 4), flatten()],
    [conv1d(400, 2 * n, 50), maxpool(4, 4), dropout(), conv1d(6, 4 * n),
     conv1d(6, 4 * n), conv1d(6, 4 * n), maxpool(2, 2), flatten()],
    [mu.Merge.Concat(axis=1), dropout()],
    [conv1d(6, n), conv1d(6, 2 * n), conv1d(6, 4 * n), dropout()],
    [mu.Merge.Sum()]], edges='1;10;011;0001;00011', name=name)
# endregion

def get_model():
  model = get_container(flatten=False)
  # add feature_extract net
  # feature_extracting_net(model, )
  model.add(feature_extracting_net('deepsleepnet'))
  model.add(BiLSTM(batch_size=th.batch_size))
  return finalize(model)

if __name__ == "__main__":
  model = get_model()
  pass
