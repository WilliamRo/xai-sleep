from tframe import mu
from tframe import tf
from tframe import context



def get_initial_model():
  from fnn_core import th

  model = mu.Classifier(mark=th.mark)

  key = '08-model'
  if not context.in_pocket(key): context.put_into_pocket('08-model', model)

  model.add(mu.Input(sample_shape=th.input_shape))

  return model



def finalize(model: mu.Classifier, flatten=False, use_gap=False):
  from fnn_core import th

  if use_gap:
    model.add(mu.GlobalAveragePooling1D())
    # model.add(mu.Flatten())
    model.add(mu.Activation('softmax'))
  else:
    if flatten: model.add(mu.Flatten())

    model.add(mu.Dense(num_neurons=th.output_dim))
    model.add(mu.Activation('softmax'))

  model.build(metric=['f1', 'accuracy'], batch_metric='accuracy')

  # Export variables
  # model.launch_model(th.overwrite)
  # from tframe import context
  # context.add_var_to_export('W', tf.trainable_variables()[-2])

  return model


