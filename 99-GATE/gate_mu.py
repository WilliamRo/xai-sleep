from tframe import Classifier
from tframe import mu
from tframe.models import Recurrent
from tframe.layers import Input, Activation
from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config

from gate_core import th


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
  # context.customized_loss_f_net = add_customized_loss_f_net
  model.build(batch_metric=['accuracy'])
  return model

def feature_extracting_net(model):
  for a in th.archi_string.split('-'):
    if a == 'm':
      model.add(mu.MaxPool1D(2, 2))
    else:
      filters = int(a)
      model.add(mu.Conv1D(filters, th.kernel_size,
                            activation=th.activation))

def get_data_fusion_model():
  model = get_container(flatten=False)
  feature_extracting_net(model)
  return finalize(model)


def get_feature_fusion_model():
  from tframe.nets.octopus import Octopus
  from gate_core import th

  model = get_container(flatten=False)
  oc: Octopus = model.add(Octopus())

  assert len(th.fusion_channels) == 3

  # Input 1
  c = len(th.fusion_channels[0])
  li = oc.init_a_limb('input-1', [3000, c])
  feature_extracting_net(li)

  # Input 2
  c = len(th.fusion_channels[1])
  li = oc.init_a_limb('input-2', [3000, c])
  feature_extracting_net(li)

  # Input 3
  c = len(th.fusion_channels[2])
  li = oc.init_a_limb('input-3', [3000, c])
  feature_extracting_net(li)

  oc.set_gates([1, 1, 1])

  return finalize(model)

def get_decision_fusion_model():
  pass


