from tframe import mu
from tframe import context

from tframe.layers.pooling import ReduceMean



def get_initial_model():
  from s2s_core import th

  model = mu.Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))

  return model



def finalize(model: mu.Classifier, flatten=False, use_gap=False):
  from s2s_core import th

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


