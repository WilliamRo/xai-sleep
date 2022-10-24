import os.path
from typing import List
from tframe.data.sequences.seq_set import SequenceSet
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from roma import console

import numpy as np


class SleepSet(SequenceSet):
    STAGE_KEY = 'STAGE'

    # region: Properties

    @property
    def signal_groups(self) -> List[SignalGroup]:
        return self.properties['signal_groups']

    # endregion: Properties

    # region: APIs

    @classmethod
    def load_as_tframe_data(cls, data_dir):
        raise NotImplementedError

    @classmethod
    def load_raw_data(cls, data_dir):
        raise NotImplementedError

    def configure(self, channel_select: str):
        """
        channel_select examples: '0,2,6'
        """
        raise NotImplementedError

    def report(self):
        raise NotImplementedError

    # endregion: APIs

    # region: Data IO

    @classmethod
    def read_edf_data_pyedflib(cls,
                               fn: str,
                               channel_list: List[str] = None,
                               freq_modifier=None,
                               length=None) -> List[DigitalSignal]:
        """Read .edf file using pyedflib package.

        :param fn: file name
        :param channel_list: list of channels. None by default.
        :param freq_modifier: This arg is for datasets such as Sleep-EDF, in which
                              frequency provided is incorrect.
        :return: a list of DigitalSignals
        """
        import pyedflib

        # Sanity check
        assert os.path.exists(fn)

        signal_dict = {}
        with pyedflib.EdfReader(fn) as file:
            # Check channels
            all_channels = file.getSignalLabels()
            if channel_list is None: channel_list = all_channels
            # Read channels
            for channel_name in channel_list:
                # Get channel id
                chn = all_channels.index(channel_name)
                frequency = file.getSampleFrequency(chn)

                # Apply freq_modifier if provided
                if callable(freq_modifier): frequency = freq_modifier(frequency)

                # Initialize an item in signal_dict if necessary
                if frequency not in signal_dict: signal_dict[frequency] = []
                # Read signal
                signal_dict[frequency].append(
                    (channel_name, file.readSignal(chn)[:length]))

        # Wrap data into DigitalSignals
        digital_signals = []
        for frequency, signal_list in signal_dict.items():
            ticks = np.arange(len(signal_list[0][1])) / frequency
            digital_signals.append(DigitalSignal(
                np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
                channel_names=[name for name, _ in signal_list],
                label=f'Freq=' f'{frequency}'))

        return digital_signals

    @classmethod
    def read_edf_anno_mne(cls, fn: str, allow_rename=True) -> list:
        from mne import read_annotations

        # Check extension
        if fn[-3:] != 'edf':
            if not allow_rename:
                # Rename .rec file if necessary, since mne package works only for
                # files with .rec extension
                raise TypeError(f'!! extension of `{fn}` is not .edf')
            os.rename(fn, fn + '.edf')
            fn = fn + '.edf'

        assert os.path.exists(fn)

        stage_anno = []
        raw_anno = read_annotations(fn)
        anno = raw_anno.to_data_frame().values
        anno_dura = anno[:, 1]
        anno_desc = anno[:, 2]
        for dura_num in range(len(anno_dura) - 1):
            for stage_num in range(int(anno_dura[dura_num]) // 30):
                stage_anno.append(anno_desc[dura_num])
        return stage_anno

    @classmethod
    def read_rrsh_data_mne(cls, fn: str, channel_list: List[str] = None,
                           start=None, end=None) -> List[DigitalSignal]:
        """Read .edf file using `mne` package"""
        from mne.io import read_raw_edf
        from mne.io.edf.edf import RawEDF

        assert os.path.exists(fn)

        signal_dict = {}
        with read_raw_edf(fn, preload=False) as raw_data:
            assert isinstance(raw_data, RawEDF)
            # resample data to 100Hz
            frequency = raw_data.info['sfreq']
            if frequency != 100:
                raw_data = raw_data.resample(100)
                frequency = raw_data.info['sfreq']
            # Check channels
            all_channels = raw_data.ch_names
            all_data = raw_data.get_data()
            if channel_list is None: channel_list = all_channels
            # Read Channels
            for channel_name in channel_list:
                chn = all_channels.index(channel_name)
                if frequency not in signal_dict: signal_dict[frequency] = []
                signal_dict[frequency].append(
                    (channel_name, all_data[chn][start:end]))

        # Wrap data into DigitalSignals
        digital_signals = []
        for frequency, signal_list in signal_dict.items():
            ticks = np.arange(len(signal_list[0][1])) / frequency
            digital_signals.append(DigitalSignal(
                np.stack([x for _, x in signal_list], axis=-1), ticks=ticks,
                channel_names=[name for name, _ in signal_list],
                label=f'Freq=' f'{frequency}'))

        return digital_signals

    @classmethod
    def read_rrsh_anno_xml(cls, fn: str, allow_rename=True) -> list:
        import xml.dom.minidom as xml

        if fn[-3:] != 'XML':
            if not allow_rename:
                # Rename .rec file if necessary, since mne package works only for
                # files with .rec extension
                raise TypeError(f'!! extension of `{fn}` is not .edf')
            os.rename(fn, fn + '.XML')
            fn = fn + '.XML'
        assert os.path.exists(fn)

        dom = xml.parse(fn)
        root = dom.documentElement
        sleep_stages = root.getElementsByTagName('SleepStage')
        stage_anno = [int(stage.firstChild.data) for stage in sleep_stages]
        stage_anno = [4 if i == 5 else i for i in stage_anno]
        stage_anno = [5 if i == 9 else i for i in stage_anno]
        return stage_anno

    # endregion: Data IO

    # region: Data Configuration

    def format_data(self):
        from xslp_core import th

        console.show_status(f'Formating data...')
        features = self.features
        targets = self.targets
        sample_length = th.random_sample_length
        if th.use_rnn:
            for i, sg_data in enumerate(features):
                len = sg_data.shape[0]
                data_reshape = sg_data.reshape(len // sample_length,
                                               sample_length)
                targets_reshape = targets[i].reshape(len // sample_length, 1)
                features[i] = data_reshape
                targets[i] = targets_reshape
        else:
            for i, sg_data in enumerate(features):
                len, chn = sg_data.shape[0], sg_data.shape[1]
                data_reshape = sg_data.reshape(len // sample_length,
                                               sample_length, chn)
                targets_reshape = targets[i].reshape(len // sample_length, 1)
                features[i] = data_reshape
                targets[i] = targets_reshape
        self.features = features
        self.targets = targets
        console.show_status(f'Finishing formating data...')
        # Set targets

    def partition(self, train_ratio, val_ratio, test_ratio):
        from xslp_core import th
        features = self.features
        targets = self.targets

        if th.use_rnn:
            person_num = int(th.data_config.split(':')[1])
            train_person = int(person_num * train_ratio)
            val_person = int(person_num * val_ratio)
            train_set_features = features[:train_person]
            train_set_targets = targets[:train_person]
            val_set_features = features[train_person:train_person + val_person]
            val_set_targets = targets[train_person:train_person + val_person]
            test_set_features = features[train_person + val_person:]
            test_set_targets = targets[train_person + val_person:]
            train_set = self.get_sequence_data(train_set_features,
                                               train_set_targets)
            val_set = self.get_sequence_data(val_set_features, val_set_targets)
            test_set = self.get_sequence_data(test_set_features,
                                              test_set_targets)

            return [train_set, val_set, test_set]
        else:
            self.features = np.vstack(features[:])
            self.targets = np.vstack(targets[:])
            self.properties[self.NUM_CLASSES] = 5
            data_sets = self.split(train_ratio, val_ratio, test_ratio,
                                   random=True,
                                   over_classes=True)
            # TODO: temporary workaround
            from tframe import DataSet
            for ds in data_sets: ds.__class__ = DataSet

            return data_sets

    def get_sequence_data(self, features: List, targets: List):
        features_list = []
        targets_list = []
        for i, feature in enumerate(features):
            nums = feature.shape[0] // 5
            for j in range(nums):
                features_list.append(feature[j * 5:(j + 1) * 5])
                targets_list.append(targets[i][j * 5:(j + 1) * 5])
        data_set = SequenceSet(features=features_list,
                               targets=targets_list,
                               name='SleepData')
        assert isinstance(data_set, SequenceSet)
        return data_set

    # endregion: Data Configuration

    # region: Visualization

    def show(self, channels: List[str] = None, **kwargs):
        from pictor import Pictor
        from pictor.plotters import Monitor

        p = Pictor(title='SleepSet', figure_size=(8, 6))
        p.objects = self.signal_groups
        p.add_plotter(Monitor(**kwargs))
        p.show()

    # endregion: Visualization
