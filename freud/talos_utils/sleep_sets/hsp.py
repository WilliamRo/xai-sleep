from collections import OrderedDict
from datetime import datetime
from freud.talos_utils.slp_set import SleepSet
from roma import console, io, Nomear
from pictor.objects.signals.signal_group import Annotation
from pictor.objects.signals.signal_group import DigitalSignal, SignalGroup
from typing import List

import os
import numpy as np
import math
import re
import traceback



class HSPSet(SleepSet):
  """Reference: https://bdsp.io/content/hsp/2.0/

  Dataset Folder Structure
  ------------------------
  The folder structure follows the BIDS (Brain Imaging Data Structure)
  specification version 1.7.0 for organizing EEG (electroencephalogram)
  data collected from multiple sites.

  There are four main levels of the folder hierarchy, these are:
  bids -> sub-ID -> ses-ID -> eeg

  Bids-root-folder/
	└── dataset_description.json
	└── participants.json
	└── participants.tsv
	└── README
	└── sub-Id/
		└── ses-01/
			└── sub-SiteIdPatientId_ses-01_scans.tsv
			└── eeg
				└── sub-Id_ses-1_task-psg_annotations.tsv
				└── sub-Id_ses-1_task-psg_channels.tsv
				└── sub-Id_ses-1_task-psg_eeg.edf
				└── sub-Id_ses-1_task-psg_eeg.json
				└── sub-Id_ses-1_task-psg_pre.csv
  """

  ANNO_LABELS = ['Sleep_stage_W', 'Sleep_stage_N1', 'Sleep_stage_N2',
                 'Sleep_stage_N3', 'Sleep_stage_R',  'Sleep_stage_?']

  EEG_EOG = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1',
             'E1-M[12]', 'E2-M[12]']

  GROUPS = [('EEG F3-M2', 'EEG C3-M2', 'EEG O1-M2',
             'EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1'),
            ('EOG E1-M2', 'EOG E2-M1',
             'EOG E1-M1', 'EOG E2-M2', )]


  @staticmethod
  def channel_map(edf_ck):
    """Map EDF channel names to standard channel names. Used in reading raw data
    """
    # For edf_ck match 'F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1'
    # Using regular expression
    if re.match(r'^[FCO][1234][\-]M[12]$', edf_ck):
      return f'EEG {edf_ck}'

    # In some cases, two EOG channels may use the same reference electrode
    # e.g., E1-M2, E2-M2 (sub-S0001111190905_ses-1)
    #       E1-M2, E2-M1 (sub-S0001111190905_ses-4)
    if re.match(r'^E[12][\-]M[12]$', edf_ck):
      return f'EOG {edf_ck}'

    return edf_ck


  @classmethod
  def load_sg_from_raw_files(cls, ses_dir, dtype=np.float16, max_sfreq=128,
                             **kwargs):
    """Convert an `.edf` file into a SignalGroup.

    Arg
    ---
    ses_dir: str, session directory
             e.g., ...\hsp_raw\sub-S0001111190905\ses-1
    """

    # (0) Check file completeness
    ho = HSPOrganization(ses_dir)
    if not os.path.exists(ho.edf_path) or not os.path.exists(ho.anno_path):
      raise FileNotFoundError(f'File not found: {ho.edf_path} or {ho.anno_path}')

    # (1) Read annotations
    annotation = cls.load_hsp_annotation(ho.anno_path)

    # (2) Read psg data as digital signals
    digital_signals: List[DigitalSignal] = cls.read_digital_signals_mne(
      ho.edf_path, dtype=dtype, max_sfreq=max_sfreq,
      chn_map=cls.channel_map, groups=cls.GROUPS)

    # (3) Wrap data into signal group
    sg = SignalGroup(digital_signals, label=ho.sg_label)
    assert len(sg.channel_names) == 8

    sg.annotations[cls.ANNO_KEY_GT_STAGE] = annotation

    # (4) Sanity check and return
    record_minus_anno = sg.total_duration - annotation.intervals[-1][1]
    sg.put_into_pocket('edf_duration-anno_duration',
                       record_minus_anno, local=True)
    return sg


  @classmethod
  def load_hsp_annotation(cls, anno_path):
    """
    Notes:
    (1) first/last epoch in csv file may not have sleep stage annotation !
    """
    import pandas as pd

    # Read intervals and annotations
    label2int = {lb: i for i, lb in enumerate(cls.ANNO_LABELS)}
    # PITFALL: 'Sleep_stage_R' and 'Sleep_stage_REM' both exist
    label2int['Sleep_stage_REM'] = label2int['Sleep_stage_R']

    intervals, annotations = [], []
    a_epochs = []

    df = pd.read_csv(anno_path)
    # PITFALL: ' 22:49:31.00' does not match format '%H:%M:%S' (sub-S0001111531526/ses-4)
    time_fmt = '%H:%M:%S'
    global_tic, anno_tic, anno_toc = None, None, None
    # Traverse rows
    for i, row in df.iterrows():
      epoch, duration, evt = row['epoch'], row['duration'], row['event']
      time_stamp = row['time']

      # global tic !
      if i == 0: global_tic = datetime.strptime(time_stamp, time_fmt)
      toc = datetime.strptime(time_stamp, time_fmt)

      # Calculate onset
      onset = (toc - global_tic).total_seconds()
      if onset < 0: onset += 24 * 3600

      # Only sleep stage event is supported for now
      if evt not in label2int:
        # if duration == 30:
        #   print(evt)   # For debug
        continue

      # Convert epoch and duration from string to int/float
      duration = float(duration)
      assert math.isclose(duration, 30, abs_tol=1e-6)

      # Record first/last time stamp and epoch with stage annotation
      if len(intervals) == 0: anno_tic = toc
      anno_toc = toc

      # Append interval and annotation
      intervals.append((onset, onset + duration))
      annotations.append(label2int[evt])

      a_epochs.append(int(epoch))

    # Sanity check
    # tic_toc_duration = (anno_toc - anno_tic).total_seconds()
    # if tic_toc_duration < 0: tic_toc_duration += 24 * 3600
    anno_duration = sum([itv[1] - itv[0] for itv in intervals])

    # PITFALL: Subject going to bathroom may cause missing epochs
    # if a_epochs[-1] - a_epochs[0] != len(a_epochs) - 1:
    #   missings = [i for i in range(a_epochs[0], a_epochs[-1] + 1)
    #               if i not in a_epochs]
    #   raise ValueError(f'Epochs are not continuous: {missings}')
    #
    # if not math.isclose(anno_duration, tic_toc_duration + 30, abs_tol=1e-6):
    #   raise ValueError(f'anno_duration ({anno_duration}) != tic_toc_duration ({tic_toc_duration})')

    return Annotation(intervals, annotations, labels=cls.ANNO_LABELS)


  @classmethod
  def convert_rawdata_to_signal_groups(
      cls, ses_folder_list: list, tgt_dir, dtype=np.float16, max_sfreq=128,
      **kwargs):
    # (0) Check target directory
    if not os.path.exists(tgt_dir): os.makedirs(tgt_dir)
    console.show_status(f'Target directory set to `{tgt_dir}` ...')

    # (0.1) Report number of files to be converted
    ho_list = [HSPOrganization(ses_folder) for ses_folder in ses_folder_list]
    sg_file_paths = [
      os.path.join(tgt_dir, ho.get_sg_file_name(dtype, max_sfreq))
      for ho in ho_list]
    convert_list = [(sg_p, ses_p)
                    for sg_p, ses_p in zip(sg_file_paths, ses_folder_list)
                    if not os.path.exists(sg_p)]
    n_total = len(ses_folder_list)
    n_convert = len(convert_list)

    console.show_status(f'{n_total - n_convert} files already converted.')
    console.show_status(f'Converting {n_convert}/{n_total} files ...')

    # (1) Convert files
    n_success = 0
    sg_job_list = [sg_p for sg_p, _ in convert_list]
    success_sg_path_list = [p for p in sg_file_paths if p not in sg_job_list]
    for i, (sg_path, ses_path) in enumerate(convert_list):
      console.show_status(f'Converting {i + 1}/{n_convert} {ses_path} ...')
      console.print_progress(i, n_convert)

      try:
        sg: SignalGroup = cls.load_sg_from_raw_files(
          ses_dir=ses_path, dtype=dtype, max_sfreq=max_sfreq)
        io.save_file(sg, sg_path, verbose=True)
        success_sg_path_list.append(sg_path)
        n_success += 1
      except Exception as e:
        # Print error message
        console.warning(
          f"Failed to convert `{ses_path}`. Full error traceback:")
        # Or traceback.print_exc() for direct output
        console.warning(traceback.format_exc())
        continue

    console.show_status(f'Successfully converted {n_success}/{n_convert} files.')

    # Return available sg paths
    return success_sg_path_list


  @classmethod
  def load_as_signal_groups(cls, src_dir, max_sfreq=128,
                            **kwargs) -> List[SignalGroup]:

    # (0) Get configs
    JUST_CONVERSION = kwargs.get('just_conversion', False)
    PREPROCESS = kwargs.get('preprocess', '')

    # TODO: Implement this

    return

    # (1) Find patient IDs
    ss_path = os.path.join(data_dir, f'mass{ssid}')
    pids = [fn[3:10] for fn in walk(
      ss_path, 'file', '*Base.edf', return_basename=True)]
    n_patients = len(pids)

    # (2) Load signal groups
    signal_groups: List[SignalGroup] = []

    for i, pid in enumerate(pids):
      # (2.1) Create a function to load raw signal group
      load_raw_sg = lambda: cls.load_as_raw_sg(
        data_dir, pid, n_patients=n_patients, i=i,
        max_sfreq=max_sfreq, **kwargs)

      # (2.2) Parse pre-process configs
      pp_configs, suffix = cls.parse_preprocess_configs(PREPROCESS)
      # TODO: currently we don't support suffix
      assert suffix == ''

      # (2.3) Try to load raw signal group
      if suffix == '':
        try:
          sg = load_raw_sg()
        except Exception as e:
          import traceback
          console.warning(f'Failed to load {pid}. Error: {e}')
          traceback.print_exc()
          continue

        if not JUST_CONVERSION: signal_groups.append(sg)
        continue

      # (2.4) TODO


      # This is for 00-data-conversion scripts
      if JUST_CONVERSION: signal_groups.clear()

    # (-1) Show status
    console.show_status(f'Successfully read {n_patients} files.')
    return signal_groups




class HSPAgent(Nomear):

  def __init__(self, meta_dir, data_dir, meta_time_stamp='20231101',
               access_point_name=None):
    self.meta_dir = meta_dir
    self.meta_time_stamp = meta_time_stamp
    self.access_point_name = access_point_name

    self.data_dir = data_dir

  # region: Properties

  @property
  def meta_path(self):
    meta_file_name = f'bdsp_psg_master_{self.meta_time_stamp}.csv'
    _meta_path = os.path.join(self.meta_dir, meta_file_name)
    assert os.path.exists(_meta_path), f'Meta data not found: {_meta_path}'
    return _meta_path

  @Nomear.property()
  def patient_dict(self):
    patient_dict_path = self.meta_path.replace('.csv', '.od')
    if os.path.exists(patient_dict_path):
      return io.load_file(patient_dict_path, verbose=True)

    od = self.generate_patient_dict(self.meta_path)
    io.save_file(od, patient_dict_path, verbose=True)
    return od

  # endregion: Properties

  # region: Public Methods

  # TODO: BETA
  def filter_patients(self, min_n_sessions=1, should_have_annotation=True,
                      return_folder_names=False):
    filtered_dict = OrderedDict()
    for pid, sess_dict in self.patient_dict.items():
      # Check annotation if required
      if should_have_annotation:
        sess_dict = {k: v for k, v in sess_dict.items() if v['has_annotations']}
      # Check session number
      if len(sess_dict) >= min_n_sessions:
        filtered_dict[pid] = sess_dict
    if return_folder_names: return self.convert_to_folder_names(filtered_dict)
    return filtered_dict

  def convert_to_folder_names(self, patient_dict: OrderedDict, local=False):
    """e.g., <APN>/bdsp-psg-access-point/PSG/bids/sub-S0001111190905/ses-1/"""
    if local: src_path = self.data_dir
    else: src_path = f'{self.access_point_name}/bdsp-psg-access-point/PSG/bids'

    assert src_path[-1] != '/'

    folder_list = []
    for pid, sess_dict in patient_dict.items():
      for sess_id, sess_info in sess_dict.items():
        folder_name = f'{src_path}/sub-{sess_info["site_id"]}{pid}/ses-{sess_id}'
        folder_list.append(folder_name)
    return folder_list

  @staticmethod
  def generate_patient_dict(meta_path) -> OrderedDict:
    import pandas as pd

    # (0) Read meta data
    df = pd.read_csv(meta_path)

    # (1) Read patients' info as a list of dictionaries
    to_boolean = {'Y': True, 'N': False}
    n_rows = df.shape[0]
    patient_dict = OrderedDict()
    for i, row in df.iterrows():
      # (1.0) Show progress
      if i == 0 or i == n_rows // 100: console.print_progress(i, n_rows)

      # (1.1) Read row
      pid = row['BDSPPatientID']
      session_id = row['SessionID']

      bids_folder = row['BidsFolder']
      site_id = row['SiteID']
      pre_sleep_questionnaire = row['PreSleepQuestionnaire']
      has_annotations = to_boolean[row['HasAnnotations']]
      has_staging = to_boolean[row['HasStaging']]
      study_type = row['StudyType']
      age = int(row['AgeAtVisit'])
      gender = row['SexDSC']

      # (1.2) Create patient slot if not exists
      if pid not in patient_dict: patient_dict[pid] = OrderedDict()
      assert session_id not in patient_dict[pid]
      patient_dict[pid][session_id] = {
        'site_id': site_id,
        'bids_folder': bids_folder,
        'pre_sleep_questionnaire': pre_sleep_questionnaire,
        'has_annotations': has_annotations,
        'has_staging': has_staging,
        'study_type': study_type,
        'age': age,
        'gender': gender,
      }

    # (2) Report progress and return
    console.show_status(f'Successfully read {len(patient_dict)} patients from'
                        f' {meta_path}')
    return patient_dict

  # region: AWS Commands

  def list_folders(self, project_folder, recursive=True):
    import subprocess

    # Define command and arguments
    command = ['aws', 's3', 'ls', project_folder]
    if recursive: command.append('--recursive')

    console.show_status(f'Executing command: {" ".join(command)}')

    # Execute the command
    result = subprocess.run(command, capture_output=True, text=True)

    # Access the output and error
    stdout = result.stdout
    stderr = result.stderr

    # Display the outputs
    if stderr: console.warning(stderr)
    else:
      console.show_status('stdout:')
      console.split()
      print(stdout)
      console.split()

  @staticmethod
  def run_command_realtime(command):
    import subprocess, sys

    # Start the process
    process = subprocess.Popen(
      command,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True,
      bufsize=1
    )

    try:
      # Read stdout and stderr in real-time
      while True:
        # Read one line at a time
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
          break
        if output:
          console.clear_line()
          sys.stdout.write(output[:-1])
          sys.stdout.flush()

    except KeyboardInterrupt:
      # If interrupted, terminate the subprocess
      process.terminate()
      print("\nProcess terminated.")
      return None  # or return an appropriate code

    # Get the return code
    return_code = process.poll()

    # Print any remaining stderr
    for line in process.stderr: console.warning(line)

    console.clear_line()

    return return_code

  def check_folder_complete(self, folder_path):
    # Check level 0
    if not os.path.exists(folder_path): return False

    ho = HSPOrganization(folder_path)

    # Check level 1
    paths = [ho.eeg_path, ho.tsv_path]
    if not all([os.path.exists(p) for p in paths]): return False

    # Check level 2
    paths = [ho.channel_path, ho.edf_path]
    if not all([os.path.exists(p) for p in paths]): return False

    return True

  def copy_a_folder(self, project_folder):
    # (0) Check if the folder already exists
    #  e.g. local_path = 'F:\\data\\hsp\\sub-S0001111190905/ses-1/'
    local_path = os.path.join(self.data_dir, project_folder.split('bids/')[-1])
    local_path = os.path.abspath(local_path)

    if self.check_folder_complete(local_path):
      console.show_status(f'Folder already exists: {local_path}')
      return 'exist'

    # (1) Download data
    # Define the command and its arguments
    command = ['aws', 's3', 'cp', project_folder, local_path, '--recursive']

    return_code = self.run_command_realtime(command)

    if return_code != 0:
      console.warning(f"Command failed with return code {return_code}")
      return 'error'
    else:
      console.show_status(f'Downloaded data to: {local_path}')
      return 'success'

  def download_folders(self, folder_list: list):
    N = len(folder_list)
    n_success, n_exist, n_error = 0, 0, 0
    for i, path in enumerate(folder_list):
      console.show_status(f'Progress: {i}/{N} ...')
      ret = self.copy_a_folder(path)

      if ret == 'exist': n_exist += 1
      elif ret == 'success': n_success += 1
      elif ret == 'error': n_error += 1
      else: raise KeyError(f'!! unknown return code `{ret}`')

    console.show_info('Summary:')
    console.supplement(f'{n_exist} folders already exist.', level=2)
    console.supplement(f'Successfully downloaded {n_success} folders.', level=2)
    console.supplement(f'Failed to downloaded {n_error} folders.', level=2)

  # endregion: AWS Commands

  # endregion: Public Methods


class HSPOrganization(object):
  """ Bids-root-folder/
      └── dataset_description.json
      └── participants.json
      └── participants.tsv
      └── README
      └── sub-Id/
        └── ses-01/
          └── sub-SiteIdPatientId_ses-01_scans.tsv
          └── eeg
            └── sub-Id_ses-1_task-psg_annotations.tsv
            └── sub-Id_ses-1_task-psg_channels.tsv
            └── sub-Id_ses-1_task-psg_eeg.edf
            └── sub-Id_ses-1_task-psg_eeg.json
            └── sub-Id_ses-1_task-psg_pre.csv
  """

  def __init__(self, ses_path):
    self.ses_path = ses_path

    self.ses_id = os.path.basename(ses_path)
    self.sub_id = os.path.basename(os.path.dirname(ses_path))

    prefix = f'{self.sub_id}_{self.ses_id}'

    # Level 1
    self.eeg_path = os.path.join(ses_path, 'eeg')
    self.tsv_path = os.path.join(ses_path, f'{prefix}_scans.tsv')

    # Level 2
    prefix = f'{self.sub_id}_{self.ses_id}_task-psg'
    self.anno_path = os.path.join(self.eeg_path, f'{prefix}_annotations.csv')
    self.channel_path = os.path.join(self.eeg_path, f'{prefix}_channels.tsv')
    self.edf_path = os.path.join(self.eeg_path, f'{prefix}_eeg.edf')
    self.json_path = os.path.join(self.eeg_path, f'{prefix}_eeg.json')
    self.pre_path = os.path.join(self.eeg_path, f'{prefix}_pre.csv')

  @property
  def sg_label(self): return f'{self.sub_id}_{self.ses_id}'

  def get_sg_file_name(self, dtype, max_sfreq):
    dtype_str = str(dtype).split('.')[-1].replace('>', '')
    dtype_str = dtype_str.replace("'", '')
    return f'{self.sg_label}({dtype_str},{max_sfreq}Hz).sg'


if __name__ == '__main__':
  import pandas as pd

  OVERWRITE = 1

  data_dir = r'../../../data/hsp'
  meta_path = os.path.join(data_dir, 'bdsp_psg_master_20231101.csv')
  patient_dict_path = os.path.join(data_dir, 'patient_dict_20231101.od')

  if os.path.exists(patient_dict_path) and not OVERWRITE:
    patient_dict = io.load_file(patient_dict_path, verbose=True)
  else:
    patient_list = HSPAgent.generate_patient_dict(meta_path)
    io.save_file(patient_list, patient_dict_path, verbose=True)

  print('Done!')


