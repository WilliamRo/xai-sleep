from tframe import Classifier
from tframe import mu

from layer.attention import Attention
from senet_core import th


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


# endregion: input and output

# region: api
def conv1d(filters, kernel_size, strides=1):
    """Conv1D layer"""
    return mu.Conv1D(filters, kernel_size, strides,
                     padding='valid',
                     use_batchnorm=th.use_batchnorm,
                     activation=th.activation)


def maxpool(pool_size, strides):
    """Maxpool layer"""
    return mu.MaxPool1D(pool_size, strides, padding='valid')


def dropout():
    """Dropout"""
    return mu.Dropout(0.5)


def flatten():
    return mu.Flatten()


def dense(output_dim):
    mu.Dense(output_dim, use_bias=False)


def activation(method):
    return mu.Activation(method)

# endregion: api

# region: model
def feature_extracting_net(name):
    return mu.ForkMergeDAG(vertices=[
        [conv1d(64, 50, 6), maxpool(8, 8), dropout()],
        [conv1d(64, 400, 50), maxpool(4, 4), dropout()],
        [mu.Merge.Concat(axis=1)]],
        edges='1;10;011', name=name)


def get_cscnn_model():
    from tframe.nets.octopus import Octopus
    from senet_core import th

    model = get_container(flatten=False)
    model.add(feature_extracting_net('feature_extraction'))
    model.add(Attention(num_neurons=64))
    return finalize(model)
# endregion: model
