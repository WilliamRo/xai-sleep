from tframe import Classifier
from tframe import mu
from tframe.models import Recurrent
from tframe.layers import Input, Activation
from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config
from layer.gconv import GatedConv1D

from gate_core import th

# region: input and output
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

def finalize_cam(model):
  assert isinstance(model, Classifier)
  # add global average pool
  model.add(mu.GlobalAveragePooling1D())
  # Build model
  model.add(mu.Dense(th.output_dim, use_bias=False, prune_frac=0.5))
  model.add(mu.Activation('softmax'))
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

# region: build model

# region: data_fusion
def get_data_fusion_model():
  model = get_container(flatten=False)
  # add feature_extract net
  feature_extracting_net(model, )
  return finalize(model)

def get_data_fusion_model_gate():
  model = get_container(flatten=False)
  # add gate
  model.add(GatedConv1D(filters=32, kernel_size=5,
                        activation=th.activation,
                        use_batchnorm=th.use_batchnorm))
  # add feature_extract net
  feature_extracting_net(model)
  return finalize(model)

def get_data_fusion_model_cam():
  from tframe import mu
  model = get_container(flatten=False)
  # add feature_extract net
  feature_extracting_net(model)
  return finalize_cam(model)
# endregion: data_fusion

# region: feature fusion
def get_feature_fusion_model():
  from tframe.nets.octopus import Octopus
  from gate_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  fusion_channels = th.fusion_channels
  assert len(fusion_channels) == 3

  # Input 1
  c = len(fusion_channels[0])
  li = oc.init_a_limb('input-1', [3000, c])
  feature_extracting_net(li)

  # Input 2
  c = len(fusion_channels[1])
  li = oc.init_a_limb('input-2', [3000, c])
  feature_extracting_net(li)

  # Input 3
  c = len(fusion_channels[2])
  li = oc.init_a_limb('input-3', [3000, c])
  feature_extracting_net(li)

  oc.set_gates([1, 1, 1])
  return finalize(model)

def get_feature_fusion_model_gate():
  from tframe.nets.octopus import Octopus
  from gate_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  fusion_channels = th.fusion_channels
  assert len(fusion_channels) == 3

  # Input 1
  c = len(fusion_channels[0])
  li = oc.init_a_limb('input-1', [3000, c])
  li.add(GatedConv1D(filters=32, kernel_size=5,
                     activation=th.activation,
                     use_batchnorm=th.use_batchnorm))
  feature_extracting_net(li)

  # Input 2
  c = len(fusion_channels[1])
  li = oc.init_a_limb('input-2', [3000, c])
  li.add(GatedConv1D(filters=32, kernel_size=5,
                     activation=th.activation,
                     use_batchnorm=th.use_batchnorm))
  feature_extracting_net(li)

  # Input 3
  c = len(fusion_channels[2])
  li = oc.init_a_limb('input-3', [3000, c])
  li.add(GatedConv1D(filters=32, kernel_size=5,
                     activation=th.activation,
                     use_batchnorm=th.use_batchnorm))
  feature_extracting_net(li)

  oc.set_gates([1, 1, 1])
  return finalize(model)

def get_feature_fusion_model_cam():
  from tframe.nets.octopus import Octopus
  from gate_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  fusion_channels = th.fusion_channels
  assert len(fusion_channels) == 3

  # Input 1
  c = len(fusion_channels[0])
  li = oc.init_a_limb('input-1', [3000, c])
  feature_extracting_net(li)

  # Input 2
  c = len(fusion_channels[1])
  li = oc.init_a_limb('input-2', [3000, c])
  feature_extracting_net(li)

  # Input 3
  c = len(fusion_channels[2])
  li = oc.init_a_limb('input-3', [3000, c])
  feature_extracting_net(li)

  oc.set_gates([1, 1, 1])
  return finalize_cam(model)
# endregion: feature fusion
def get_decision_fusion_model():
  pass
# endregion: build model

