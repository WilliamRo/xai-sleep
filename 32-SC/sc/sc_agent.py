from roma import console
from roma import finder
from roma import io
from roma import Nomear

import os
import pandas as pd
import re



class SCAgent(Nomear):

  DEFAULT_PATTERN = '*(trim1800;128).sg'
  DEFAULT_FP_FILENAME = '0211-N153-ch(all)-F(20)-A(128).fp'

  def __init__(self):
    self.edfx_root = self.get_edfx_root()
    self.sc_subjects_xls_path = os.path.join(self.edfx_root, 'SC-subjects.xls')

  # region: Properties

  # region: Data Frames

  @Nomear.property()
  def raw_patient_info_df(self):
    return pd.read_excel(self.sc_subjects_xls_path)

  @Nomear.property()
  def all_subjects_list(self):
    return self.raw_patient_info_df['subject'].unique().tolist()

  @Nomear.property()
  def valid_subject_list(self):
    df = self.raw_patient_info_df
    all_subjects = self.all_subjects_list
    return [s for s in all_subjects if len(df[df['subject'] == s]) > 1]

  @Nomear.property()
  def invalid_subject_list(self):
    all_subjects = self.all_subjects_list
    valid_subjects = self.valid_subject_list
    return sorted(list(set(all_subjects) - set(valid_subjects)))

  @Nomear.property()
  def patient_info_df(self):
    pinfo = self.raw_patient_info_df[['subject', 'age', 'sex (F=1)']]
    pinfo = pinfo.rename(columns={'sex (F=1)': 'sex'})
    pinfo = pinfo[pinfo['subject'].isin(self.valid_subject_list)]
    pinfo = pinfo.drop_duplicates()
    pinfo.loc[pinfo['sex'] == 1, 'sex'] = 'F'
    pinfo.loc[pinfo['sex'] == 2, 'sex'] = 'M'
    return pinfo

  # endregion: Data Frames

  # region: SG files

  @Nomear.property()
  def all_sg_files(self):
    data_dir = os.path.join(self.edfx_root, 'sleep-cassette/')
    return finder.walk(data_dir, pattern=self.DEFAULT_PATTERN)

  @Nomear.property()
  def valid_sg_files(self):
    return [f for f in self.all_sg_files
            if int(re.search(r'SC4(\d+)[12]', f).group(1))
            in self.valid_subject_list]

  # endregion: SG files

  # region: Finger-print version 1

  def get_fp_v1_dict(self, fp_file_name=DEFAULT_FP_FILENAME,
                     return_valid=True) -> dict:
    from tframe.utils.file_tools.io_utils import load
    fp_file_path = os.path.join(self.fp_feature_root, fp_file_name)
    fps = load(fp_file_path)
    if return_valid:
      pids, _1, _2 = fps['meta']
      pids = [pid for pid in pids if int(pid[3:5]) in self.valid_subject_list]
      fps['meta'] = (pids, _1, _2)

    return fps

  # endregion: Finger-print version 1

  # endregion: Properties

  # region: IO

  @Nomear.property()
  def fp_feature_root(self):
    ROOT = os.path.abspath(__file__)
    for _ in range(2): ROOT = os.path.dirname(ROOT)
    return os.path.join(ROOT, 'features/')

  @classmethod
  def get_edfx_root(cls, dir_depth=3):
    ROOT = os.path.abspath(__file__)
    for _ in range(dir_depth): ROOT = os.path.dirname(ROOT)
    return os.path.join(ROOT, 'data/sleep-edf-database-expanded-1.0.0/')

  # endregion: IO

  # region: Public Methods

  # region: Patient Info

  def report_data_info(self):
    console.show_status('SC-subjects information:')
    console.supplement(f'Total subject #: {len(self.all_subjects_list)}')
    console.supplement(f'Valid subject #: {len(self.valid_subject_list)}')
    console.supplement(f'Invalid subject IDs: {self.invalid_subject_list}')

  def show_violin_plot(self):
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(rc={'figure.figsize': (5, 6)})
    sns.violinplot(data=self.patient_info_df, y='age', hue='sex', split=True,
                   fill=False, inner='quart', palette='muted')
    plt.tight_layout()
    plt.show()

  # endregion: Patient Info

  # region: Signal Groups

  def visualize_signal_groups(self, N=20):
    """Raw data should be preprocessed using `00-data_converter.py` script.
    Use `:ta` to toggle sleep-stage annotations
    """
    from pictor.objects.signals.signal_group import SignalGroup
    from freud.gui.freud_gui import Freud

    sg_file_list = self.valid_sg_files
    console.show_status(f'Found altogether {len(sg_file_list)} signal groups')
    sg_file_list = sg_file_list[:N]

    signal_groups = []
    for path in sg_file_list[:N]:
      sg: SignalGroup = io.load_file(path, verbose=True)
      signal_groups.append(sg)

    Freud.visualize_signal_groups(signal_groups, 'SleepEDFx-SC',
                                  default_win_duration=9999999)

  # endregion: Signal Groups

  # region: Finger-print version 1

  def visualize_fp_v1(self, fp_path=DEFAULT_FP_FILENAME):
    """FPS_v1 is generated by `01-sc-fp-generator.py` script"""
    from sc.fp_viewer import FPViewer

    fps = self.get_fp_v1_dict(fp_path)
    pfv = FPViewer(walker_results=fps)
    pfv.show()

  # endregion: Finger-print version 1

  # endregion: Public Methods



if __name__ == '__main__':
  sca = SCAgent()
  sca.report_data_info()

  # sca.show_violin_plot()
  # sca.visualize_signal_groups()
  sca.visualize_fp_v1()
