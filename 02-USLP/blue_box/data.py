from xsleep.slp_set import SleepSet
from slp_datasets.sleepedfx import SleepEDFx
from xsleep.slp_set import SleepSet
from uslp_core import th

import numpy as np
import matplotlib.pyplot as plt
import os
import pyedflib
import mne


channel = ['EEG Fpz-Cz', 'EOG horizontal']

def read_edf_mne(file_path):
    edf_file = mne.io.read_raw_edf(file_path)
    data = edf_file.get_data()
    return data


def read_npy_file(file_path):
    file = np.load(file_path)
    return file


def write_edf_pyedflib(new_file, edf_file, npy_file):
    eog_data = read_edf_mne(edf_file)[2][0: 7818000]
    eeg_data = read_npy_file(npy_file)[0]
    file = pyedflib.EdfWriter(new_file, n_channels=len(channel),)
    channel_info = []
    data_list = []

    ch_dict = {
        'label': channel[0],
        'sample_rate': 100,
        'dimension': 'V',
    }
    channel_info.append(ch_dict)
    data_list.append(eeg_data)
    # data_list.append(np.random.normal(size=600 * 200) * 10e-5)

    ch_dict = {
        'label': channel[1],
        'sample_rate': 100,
        'dimension': 'V'
    }
    channel_info.append(ch_dict)
    data_list.append(eog_data)
    # data_list.append(np.random.normal(size=600 * 200) * 10e-5)

    file.setSignalHeaders(channel_info)
    file.writeSamples(data_list)
    file.close()

def write_edf_mne(new_file, edf_file, npy_file):
    eog_data = read_edf_mne(edf_file)[2][0: 7818000]
    eeg_data = read_npy_file(npy_file)[0]
    data = np.array([eeg_data, eog_data])

    """
    创建info结构,
    内容包括：通道名称和通道类型
    设置采样频率为:sfreq=100
    """
    info = mne.create_info(
        channel,
        ch_types=['eeg', 'eog'],
        sfreq=100
    )

    custom_raw = mne.io.RawArray(data, info)
    mne.export.export_raw(new_file, custom_raw, 'edf')

def get_annotation():
    ANNO_LABELS = ['Sleep stage W', 'Sleep stage 1', 'Sleep stage 2',
                   'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R',
                   'Movement time', 'Sleep stage ?']
    anno = SleepSet.read_annotations_mne(h_path, ANNO_LABELS)
    label1 = anno.annotations
    interval1 = anno.intervals
    for i, time in enumerate(interval1):
        if time[1] > 78180:
            interval1[i] = (interval1[i][0], 78180.0)
            label1 = label1[:i + 1]
            interval1 = interval1[:i + 1]
            break
    annotation = []
    for index, interval in enumerate(interval1):
        annotation.extend(np.ones(int(interval[1] - interval[0]) // 30, dtype=np.int) * label1[index])
    return annotation

def cal_accuracy():
    # get annotion
    ground_truth = np.array(get_annotation()[: 2606])
    predict = read_npy_file(predict_path)[: 2606]
    # calculate accuracy
    error = np.count_nonzero(ground_truth - predict)
    accuracy = 1 - error / ground_truth.shape[0]
    print(accuracy)


if __name__ == '__main__':
    generate_edf = True
    get_accuracy = False
    compare_edf_data = True

    #prepare file name
    data_config = 'sleepedfx:20:0,2'
    data_name, data_num, channel_select = data_config.split(':')
    f_name = 'SC4001E0-PSG.edf'
    h_name = 'SC4001EC-Hypnogram.edf'
    new_name = 'SC4001E0-PSG-NEW.edf'
    npy_name = 'Sleep_Fpz_Cz.npy'
    predict_name = 'SC4001E0-PSG_hypnogram.npy'

    # prepare data_path
    data_dir = os.path.join(th.data_dir, 'peiyan')
    f_path = os.path.join(data_dir, f_name)
    h_path = os.path.join(data_dir, h_name)
    new_path = os.path.join(data_dir, new_name)
    npy_path = os.path.join(data_dir, npy_name)
    predict_path = os.path.join(data_dir, predict_name)

    # generate .edf file
    if generate_edf:
        # write_edf_pyedflib(new_path, f_path, npy_path)
        write_edf_mne(new_file=new_path, edf_file=f_path, npy_file=npy_path)
    # get accuracy
    if get_accuracy:
        cal_accuracy()





