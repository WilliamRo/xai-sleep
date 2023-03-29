from tframe import Classifier, context
from tframe import mu, tf
from gate_core import th
from layer.gconv import GatedConv1D
from layer.cam_gap import GlobalAveragePooling1D
from layer.cam_conv1d import Conv1D
from layer.cam_dense import Dense


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
  # Build model
  model.add(Dense(th.output_dim, use_bias=False, prune_frac=0.5))
  model.add(mu.Activation('softmax'))
  model.build(batch_metric=['accuracy'])

  # Export variables
  model.launch_model(th.overwrite)
  context.add_var_to_export('dense', tf.trainable_variables()[-1])
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

def feature_extracting_net_cam(model, last_conv_name='last_conv_layer'):
  for index, a in enumerate(th.archi_string.split('-')):
    if a == 'm':
      model.add(mu.MaxPool1D(th.kernel_size, th.kernel_size))
    else:
      filters = int(a)
      conv_layer = conv1d(filters, th.kernel_size)
      if index == len(th.archi_string.split('-')) - 1:
        conv_layer = Conv1D(filters, th.kernel_size, strides=1,
                            use_batchnorm=th.use_batchnorm,
                            activation=th.activation,
                            name=last_conv_name)
      model.add(conv_layer)

# endregion: api

# region: build model

# region: data_fusion
def get_data_fusion_model():
  model = get_container(flatten=False)
  # add feature_extract net
  feature_extracting_net(model)
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
  feature_extracting_net_cam(model)
  # add global average pool
  model.add(GlobalAveragePooling1D(layer_name='gap_layer'))
  return finalize_cam(model)

# endregion: data_fusion

# region: feature fusion
def get_feature_fusion_model():
  from tframe.nets.octopus import Octopus
  from gate_core import th

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


def get_feature_fusion_model_gate():
  from tframe.nets.octopus import Octopus
  from gate_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  fusion_channels = th.fusion_channels()
  assert len(fusion_channels) >= 2

  for index, fusion_channel in enumerate(fusion_channels):
    # Input
    c = len(fusion_channel)
    li = oc.init_a_limb(f'input-{index + 1}', [3000, c])
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

  fusion_channels = th.fusion_channels()
  assert len(fusion_channels) >= 2

  for index, fusion_channel in enumerate(fusion_channels):
    # Input
    c = len(fusion_channel)
    li = oc.init_a_limb(f'input-{index + 1}', [3000, c])
    feature_extracting_net_cam(li, f'last_conv_layer{index + 1}')

  oc.set_gates([1, 1, 1])
  return finalize(model)


# endregion: feature fusion
def get_decision_fusion_model():
  pass
# endregion: build model
