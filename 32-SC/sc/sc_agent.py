from roma import console
from roma import finder
from roma import io
from roma import Nomear
from scipy import stats

import numpy as np
import os
import pandas as pd
import re




class SCAgent(Nomear):
  STAGES = ['W', 'N1', 'N2', 'N3', 'R']
  DEFAULT_PATTERN = '*(trim1800;128).sg'
  DEFAULT_FP_FILENAME = '0211-N153-ch(all)-F(20)-A(128).fp'

  def __init__(self, data_key='dual'):
    self.data_key = data_key
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
    sub_list = all_subjects
    if 'dual' in self.data_key:
      sub_list = [s for s in sub_list if len(df[df['subject'] == s]) > 1]
    if 'alpha' in self.data_key:
      INVA_SUBS = [11, 22, 23, 40, 51, 57, 59, 76, ]
      sub_list = [s for s in sub_list if s not in INVA_SUBS]
    return sub_list

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

  def normalize_fp(self, fps: dict):

    class Tom(Nomear):
      @Nomear.property()
      def data_dict(self): return {}

      @Nomear.property()
      def stat_dict(self):
        r = {}
        for k, v in self.data_dict.items():
          v = np.concatenate(v)
          r[k] = (np.mean(v), np.std(v))
        return r

      def reg(self, ch, dim_key, arg_name, arg, array: np.ndarray):
        key = (ch, dim_key, arg_name, arg)
        if key not in self.data_dict: self.data_dict[key] = []
        self.data_dict[key].append(array)

      def normalize(self, ch, dim_key, arg_name, arg, data: dict):
        key = (ch, dim_key, arg_name, arg)
        stages = list(data.keys())
        for s in stages:
          mu, sigma = self.stat_dict[key]
          data[s] = (data[s] - mu) / sigma

    tom = Tom()

    # Calculate mu and sigma
    for key, stage_dict in fps.items():
      if key == 'meta': continue
      _, chn, (dim_key, arg_name, arg) = key
      for array in stage_dict.values():
        tom.reg(chn, dim_key, arg_name, arg, array)

    # Normalize data
    for key, stage_dict in fps.items():
      if key == 'meta': continue
      _, chn, (dim_key, arg_name, arg) = key
      tom.normalize(chn, dim_key, arg_name, arg, stage_dict)

  def get_fp_v1_dict(self, fp_file_name=DEFAULT_FP_FILENAME,
                     normalize=True, return_valid=True) -> dict:
    from roma import io
    fp_file_path = os.path.join(self.fp_feature_root, fp_file_name)
    fps = io.load_file(fp_file_path)

    # Normalize fps if required
    if normalize: self.normalize_fp(fps)

    if return_valid:
      pids, _1, _2 = fps['meta']
      pids = [pid for pid in pids if int(pid[3:5]) in self.valid_subject_list]
      fps['meta'] = (pids, _1, _2)

    return fps

  # endregion: Finger-print version 1

  # region: Process

  @Nomear.property()
  def beta_fp_dict(self):
    return self.get_fp_v1_dict(normalize=True, return_valid=True)

  @Nomear.property()
  def beta_subject_ids(self):
    return self.beta_fp_dict['meta'][0]

  @Nomear.property()
  def beta_uni_subs(self):
    return sorted(list(set([s[:-2] for s in self.beta_subject_ids])))

  # endregion: Process

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

  def visualize_fp_v1(self, fp_path=DEFAULT_FP_FILENAME, **configs):
    """FPS_v1 is generated by `01+sc-fp-generator.py` script"""
    from sc.fp_viewer import FPViewer

    fps = self.get_fp_v1_dict(fp_path, normalize=False)
    fpv = FPViewer(walker_results=fps)
    for k, v in configs.items(): fpv.plotters[0].set(k, v)
    fpv.show()

  # endregion: Finger-print version 1

  # region: Features

  def extract_kde_vector(self, pid, channel, dim1_tuple, dim2_tuple):
    MIN_SIZE = 5
    kde_dict = {}
    xs = self.beta_fp_dict[(pid, channel, dim1_tuple)]
    ys = self.beta_fp_dict[(pid, channel, dim2_tuple)]
    for stage in self.STAGES:
      if stage not in xs: continue
      x, y = xs[stage], ys[stage]
      if len(x) < MIN_SIZE: continue
      values = np.vstack([x, y])
      kernel = stats.gaussian_kde(values)
      kde_dict[stage] = kernel

    return kde_dict

  def get_kde_dicts(self, channel, dim1_tuple, dim2_tuple):
    kde_dicts_1, kde_dicts_2 = {}, {}

    for i, s in enumerate(self.beta_subject_ids):
      # console.print_progress(i, len(subjects))
      kde_dict = self.extract_kde_vector(s, channel, dim1_tuple, dim2_tuple)
      # Put results into corresponding dict
      pid = s[:-2]
      # assert pid in uni_subs
      tgt_dict = kde_dicts_2 if pid in kde_dicts_1 else kde_dicts_1
      tgt_dict[pid] = kde_dict

    return kde_dicts_1, kde_dicts_2

  def pyhypnomics(self, knl_dict: dict, version='v1'):
    # [IMPORTANT] Wake stage should be excluded, (acc=34 -> 37)
    N = sum([knl.dataset.shape[1] for s, knl in knl_dict.items() if s != 'W'])
    def v1():
      v = []
      for stage in self.STAGES:
        if stage not in knl_dict:
          v.extend([0.] * 6)
          continue
        # Get data
        knl: stats.gaussian_kde = knl_dict[stage]
        data = knl.dataset
        # [1] Percentage
        # if stage == 'W': v.append(0.)
        # else: v.append(data.shape[1] / N)
        v.append(data.shape[1] / N)
        # [2, 3] Means
        v.extend([np.mean(data[0]), np.mean(data[1])])
        # [4, 5] Vars
        v.extend([knl.covariance[0, 0], knl.covariance[1, 1]])
        # [6] Cov
        v.append(knl.covariance[0, 1])

      return np.array(v)

    def v2():
      v = []
      data = {s: k.dataset for s, k in knl_dict.items()}
      covs = {s: [k.covariance[i, j] for (i, j) in [(0, 0), (0, 1), (1, 1)]]
              for s, k in knl_dict.items()}
      ADD_COV = 0

      # N2 as origin
      n2_mu = np.mean(data['N2'], axis=1)
      v.append(data['N2'].shape[1] / N)
      if ADD_COV: v.extend(covs['N2'])

      weights = {'W': 0.2, 'N1': 0.5, 'N3': 1, 'R': 1}

      for s, w in weights.items():
        if s not in data:
          v.extend([0.] * 3)
          if ADD_COV: v.extend([0.] * 3)
          continue
        mu = np.mean(data[s], axis=1)
        diff = (mu - n2_mu) * w
        v.append(data[s].shape[1] / N * w)
        v.extend(diff)
        if ADD_COV: v.extend(covs[s])

      return np.array(v)

    return {'v1': v1, 'v2': v2}[version]()

  def get_feature_dicts(self, channel, dim1_tuple, dim2_tuple, version='v1'):
    if not isinstance(channel, (list, tuple)): channel = [channel]
    fea_dicts_1, fea_dicts_2 = {}, {}
    for ch in channel:
      kde_dicts_1, kde_dicts_2 = self.get_kde_dicts(ch, dim1_tuple, dim2_tuple)
      for s in kde_dicts_1.keys():
        f1 = self.pyhypnomics(kde_dicts_1[s], version)
        f2 = self.pyhypnomics(kde_dicts_2[s], version)
        if s not in fea_dicts_1:
          fea_dicts_1[s] = f1
          fea_dicts_2[s] = f2
        else:
          fea_dicts_1[s] = np.concatenate([fea_dicts_1[s], f1])
          fea_dicts_2[s] = np.concatenate([fea_dicts_2[s], f2])

    return fea_dicts_1, fea_dicts_2

  def get_feature_targets(self, channel, dim1_tuple, dim2_tuple, version):
    fea_dicts_1, fea_dicts_2 = self.get_feature_dicts(
      channel, dim1_tuple, dim2_tuple,version=version)
    N = len(fea_dicts_1)
    features, targets, pids = [], [], []
    for i, pid in enumerate(fea_dicts_1.keys()):
      features.append(fea_dicts_1[pid])
      features.append(fea_dicts_2[pid])
      targets.append(i)
      targets.append(i)

      pid = pid[3:]
      pids.append(pid)
      pids.append(pid)
    return features, targets, pids

  def get_distance_matrix(self, channel, dim1_tuple, dim2_tuple, N=999,
                          method='euclidean'):
    fea_dicts_1, fea_dicts_2 = self.get_feature_dicts(
      channel, dim1_tuple, dim2_tuple)
    uni_subs = self.beta_uni_subs[:N]
    matrix = np.zeros((N, N))
    for i, s1 in enumerate(uni_subs):
      for j, s2 in enumerate(uni_subs):
        if method == 'euclidean':
          matrix[i, j] = np.linalg.norm(fea_dicts_1[s1] - fea_dicts_2[s2])
        else: raise NotImplementedError

    return matrix


  # endregion: Features

  # endregion: Public Methods



if __name__ == '__main__':
  sca = SCAgent()
  sca.report_data_info()

  # sca.show_violin_plot()
  # sca.visualize_signal_groups()
  sca.visualize_fp_v1()
