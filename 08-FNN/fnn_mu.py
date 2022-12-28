from tframe import mu



def get_initial_model():
  from fnn_core import th

  model = mu.Classifier(mark=th.mark)
  model.add(mu.Input(sample_shape=th.input_shape))
  return model



def finalize(model: mu.Classifier, flatten=False):
  from fnn_core import th

  if flatten: model.add(mu.Flatten())

  model.add(mu.Dense(num_neurons=th.output_dim))
  model.add(mu.Activation('softmax'))

  model.build(metric='accuracy', batch_metric='accuracy')
  return model


