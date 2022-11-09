from tframe import Classifier
from tframe import mu

from dsn_core import th
from tframe.layers import Activation

from tframe.layers.hyper.dense import Dense
from tframe.configs.config_base import Config



def init_model(flatten=False):
    model = Classifier(mark=th.mark)
    model.add(mu.Input(sample_shape=th.input_shape))
    if flatten: model.add(mu.Flatten())
    return model


def output_and_build(model):
    assert isinstance(model, Classifier)
    assert isinstance(th, Config)
    # Add output layer
    model.add(Dense(num_neurons=th.output_dim))
    model.add(Activation('softmax'))
    # Build model and return
    model.build(metric='accuracy', batch_metric='accuracy')
    return model

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

def feature_extracting_net(name, n=32):
    return mu.ForkMergeDAG(vertices=[
        [conv1d(50, 2*n, 6), maxpool(8, 8), dropout(), conv1d(8, 4*n),
         conv1d(8, 4*n), conv1d(8, 4*n), maxpool(4, 4), flatten()],
        [conv1d(400, 2*n, 50), maxpool(4, 4), dropout(), conv1d(6, 4*n),
         conv1d(6, 4*n), conv1d(6, 4*n), maxpool(2, 2), flatten()],
        [mu.Merge.Concat(axis=1), dropout()]],
        edges='1;10;011', name=name)

def dsn_net(name, n=32):
    return mu.ForkMergeDAG(vertices=[
        [conv1d(50, 2*n, 6), maxpool(8, 8), dropout(), conv1d(8, 4*n),
         conv1d(8, 4*n), conv1d(8, 4*n), maxpool(4, 4), flatten()],
        [conv1d(400, 2*n, 50), maxpool(4, 4), dropout(), conv1d(6, 4*n),
         conv1d(6, 4*n), conv1d(6, 4*n), maxpool(2, 2), flatten()],
        [mu.Merge.Concat(axis=1), dropout()],
        [conv1d(6, n), conv1d(6, 2*n), conv1d(6, 4*n), dropout()],
        [mu.Merge.Sum()]], edges='1;10;011;0001;00011', name=name)


def get_model():
    model = init_model()
    model.add(feature_extracting_net('feature_extraction'))
    return output_and_build(model)

