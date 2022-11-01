from tframe.data.base_classes import DataAgent
from tframe import console
from xsleep.slp_set import SleepSet

import os
import pickle
import tframe.utils.misc as misc


class SLPAgent(DataAgent):
    """To use this class, th.data_config must be specified.
    The syntax is `<...>:<...>`, e.g.,
    """

    # region: Data loading

    @classmethod
    def load(cls, **kwargs):
        """th.data_config syntax:
            data_name:data_num
        """
        th = kwargs.get('th', None)

        # Find xai-sleep/data/...
        dataset_name, data_num, channel_select = th.data_config.split(':')
        suffix = '' if dataset_name == 'rrsh' else '-alpha'
        person_num = '(all)' if data_num == '' else f'({data_num})'
        prefix = dataset_name + person_num
        model_architecture = 'fnn'
        if th.use_rnn:
            model_architecture = 'rnn'
        tfd_format_path = os.path.join(th.data_dir, dataset_name,
                                       f'{prefix}-format-{model_architecture}.tfds')
        if os.path.exists(tfd_format_path):
            with open(tfd_format_path, 'rb') as _input_:
                console.show_status('Loading `{}` ...'.format(tfd_format_path))
                dataset = pickle.load(_input_)
        else:
            tfd_config_path = os.path.join(th.data_dir, dataset_name,
                                           f'{prefix}-config-{model_architecture}.tfds')
            if os.path.exists(tfd_config_path):
                with open(tfd_config_path, 'rb') as _input_:
                    console.show_status(f'loading {tfd_config_path}...')
                    dataset = pickle.load(_input_)
            else:
                dataset = cls.load_as_tframe_data(data_dir=th.data_dir,
                                                  data_name=dataset_name,
                                                  first_k=data_num,
                                                  suffix=suffix)

                # Configure dataset (put this block into right position)
                configure = kwargs.get('configure', None)
                if callable(configure):
                    dataset = configure(dataset)
                else:
                    dataset.configure(th=th, channel_select=channel_select)
                dataset.save(tfd_config_path)
            dataset.format_data()
            dataset.save(tfd_format_path)
            console.show_status(f'Saving {tfd_format_path} ...')

        # Convert dense label to one-hot
        dense_labels, n_classes = dataset.targets, 5
        if isinstance(dense_labels, list):
            dataset.targets = [misc.convert_to_one_hot(lb, n_classes)
                               for lb in dense_labels]
        else:
            dataset.targets = misc.convert_to_one_hot(dense_labels, n_classes)

        dataset.properties['CLASSES'] = ['W', 'N1', 'N2', 'N3', 'R']
        train_set, val_set, test_set = dataset.partition(0.7, 0.1, 0.2, th=th)
        console.show_status(
            'Finishing split dataset to (train_set, val_set, test_set)...')

        return train_set, val_set, test_set

    @classmethod
    def load_as_tframe_data(cls, data_dir, data_name=None,
                            **kwargs) -> SleepSet:
        """Return: an instance of SleepSet whose `properties` attribute contains
           {'signal_groups': [<a list of SignalGroup>]}

          e.g., data_set = SleepEDFx(name=f'Sleep-EDF-Expanded{suffix_k}',
                                     signal_groups=signal_groups)
        """
        if data_name in ['sleepedf', 'physionet_sleep']:
            from xsleep.slp_datasets.sleepedfx import SleepEDFx as DataSet
        elif data_name == 'ucddb':
            from xsleep.slp_datasets.ucddb import UCDDB as DataSet
        elif data_name == 'rrsh':
            from xsleep.slp_datasets.rrsh import RRSHSet as DataSet
        else:
            raise KeyError(f'!! Unknown dataset `{data_name}`')

        data_set = DataSet.load_as_tframe_data(data_dir, data_name, **kwargs)
        return data_set

    @classmethod
    def _get_tfd_file_path(cls, data_dir, data_name, **kwargs):
        suffix = kwargs['suffix']
        return os.path.join(data_dir, data_name, f'{data_name}{suffix}.tfds')

    # endregion: Data Loading

    # region: Model evaluation

    @staticmethod
    def evaluate_model(model, data_set, batch_size=100, **kwargs):
        from tframe import Classifier
        from tframe import DataSet
        from rrsh import RRSHSet
        from sleepedfx import SleepEDFx
        from collections import Counter
        import numpy as np

        assert isinstance(model, Classifier)

        data_set.features = np.vstack(data_set.features[:])
        data_set.targets = np.vstack(data_set.targets[:])
        data_set.targets = misc.convert_to_one_hot(data_set.targets, 5)
        data_set.properties['CLASSES'] = ['W', 'N1', 'N2', 'N3', 'R']
        data_set.properties[data_set.NUM_CLASSES] = 5
        data_set.__class__ = DataSet
        model.evaluate_pro(data_set, batch_size=batch_size, verbose=True,
                           cell_width=4, show_confusion_matrix=True,
                           show_class_detail=True)

        show_in_monitor = kwargs.get('show_in_monitor', None)
        batch_size = kwargs.get('batch_size', None)
        th = kwargs.get('th', None)

        if show_in_monitor:
            th.predictions = model.classify(data_set, batch_size)
            data_set_name = th.data_config.split(':')[0]
            if data_set_name == 'rrsh':
                data_set.__class__ = RRSHSet
            elif data_set_name == 'sleepedf':
                data_set.__class__ = SleepEDFx
            data_set.show(th=th)
    # endregion: Model evaluation


if __name__ == '__main__':
    from xslp_core import th

    th.data_config = 'sleepedf:10:0,1,2'
    dataset = SLPAgent.load_as_tframe_data(th.data_dir, suffix='-alpha')
    print()
