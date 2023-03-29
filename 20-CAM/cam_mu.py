from tframe import Classifier, context
from tframe import mu, tf
from cam_core import th


# region: input and output
def get_container(flatten=False):
  model = Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  if th.centralize_data: model.add(mu.Normalize(mu=th.data_mean))
  if flatten: model.add(mu.Flatten())
  return model


def finalize(model):
  assert isinstance(model, Classifier)
  model.add(mu.Dense(th.output_dim, use_bias=False, prune_frac=0.5))
  model.add(mu.Activation('softmax'))
  # Build model
  model.build(batch_metric=['accuracy'])
  return model

# endregion: input and output

# region: api
def conv1d(filters, kernel_size, strides=1):
  """Conv1D layer"""
  return mu.Conv1D(filters, kernel_size, strides,
                   use_batchnorm=th.use_batchnorm,
                   activation=th.activation)


def feature_extracting_net(model):
  for a in th.archi_string.split('-'):
    if a == 'm':
      model.add(mu.MaxPool1D(th.kernel_size, th.kernel_size))
    else:
      filters = int(a)
      model.add(conv1d(filters, th.kernel_size))

# endregion: api

# region: data_fusion
def get_data_fusion_model():
  model = get_container(flatten=False)
  # add feature_extract net
  feature_extracting_net(model)
  model.add(mu.GlobalAveragePooling1D())
  return finalize(model)
# endregion: data_fusion

# region: feature fusion
def get_feature_fusion_model():
  from tframe.nets.octopus import Octopus
  from cam_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  fusion_channels = th.fusion_channels()
  assert len(fusion_channels) >= 2

  for index, fusion_channel in enumerate(fusion_channels):
    # Input
    c = len(fusion_channel)
    li = oc.init_a_limb(f'input-{index + 1}', [3000, c])
    feature_extracting_net(li)

  oc.set_gates([1, 1, 1])
  return finalize(model)


