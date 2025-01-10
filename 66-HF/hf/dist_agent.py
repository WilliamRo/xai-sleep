from collections import OrderedDict
from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from roma import Nomear, io, console
from pictor.xomics.omix import Omix

import time
import numpy as np
import os



class DistanceAgent(Nomear):
  """da.buffer[(model_key, conditional, time_resolution, ck, pk)] = {
      (label_1, label_2): distance_value
  }
  """

  def __init__(self, name, work_dir, cloud_dir, nights_1=None, nights_2=None):
    self.name = name

    self.work_dir = work_dir
    self.cloud_dir = cloud_dir

    self.nights_1 = nights_1
    self.nights_2 = nights_2

    self._finalize_init()

  # region: Properties

  @property
  def buffer_path(self):
    buffer_fn = f'{self.name}.da'
    buffer_path = os.path.join(self.work_dir, buffer_fn)
    return buffer_path

  @Nomear.property()
  def buffer(self) -> OrderedDict:
    if os.path.exists(self.buffer_path):
      return io.load_file(self.buffer_path, verbose=True)
    return OrderedDict()

  @property
  def sg_labels(self): return list(self.nights_1) + list(self.nights_2)

  @Nomear.property()
  def pairs_names_targets(self):
    sample_names, targets, label_pairs = [], [], []
    for i, lb1 in enumerate(self.nights_1):
      for j, lb2 in enumerate(self.nights_2):
        label_pairs.append((lb1, lb2))
        sample_names.append(f'({lb1}, {lb2})')
        targets.append(int(i == j))
    return label_pairs, sample_names, targets

  # endregion: Properties

  # region: Public Methods

  # region: - Distance Calculation

  def calculate_distance(self, time_resolution, channels, probe_keys,
                         conditional=True, overwrite=False, label_pairs=None,
                         model_key='model_1', compensate_shift=True):
    conditional = int(conditional)

    # (1) Load nebula
    freud = Freud(self.cloud_dir)

    if self.nights_1 is not None and self.nights_2 is not None:
      sg_labels = self.sg_labels
    else:
      assert isinstance(label_pairs, (list, tuple)) and len(label_pairs) > 0
      sg_labels = []
      for lb1, lb2 in label_pairs: sg_labels.extend([lb1, lb2])
      sg_labels = list(set(sg_labels))
      console.show_status(
        f'Nights list not provided, loading nebula from {len(sg_labels)} subjects.',
        prompt='[DA]')

    nebula = freud.load_nebula(sg_labels=sg_labels,
                               channels=channels,
                               time_resolution=time_resolution,
                               probe_keys=probe_keys)

    # (1.1) Confirm label pairs
    if label_pairs is None:
      label_pairs = [(l1, l2) for l1 in self.nights_1 for l2 in self.nights_2]

    # (2) Calculate distance
    assert model_key == 'model_1'
    from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
    hm = HypnoModel1()

    chn_prob_keys = [(ck, pk) for ck in channels for pk in probe_keys]
    for ck, pk in chn_prob_keys:
      # (2.1) Get distance dictionary
      buffer_key = self._get_buffer_key(
        model_key, conditional, time_resolution, ck, pk, compensate_shift)

      if buffer_key not in self.buffer: self.buffer[buffer_key] = {}
      dist_dict = self.buffer[buffer_key]

      # (2.2) Find remaining label pairs
      pairs = [p for p in label_pairs if p not in dist_dict]
      if overwrite: pairs = label_pairs
      if len(pairs) == 0:
        console.show_status(f'All distances calculated for `{buffer_key}`')
        continue

      # (2.3) Calculate distances
      console.show_status(f'Calculating distances for `{buffer_key}` ...')
      tic = time.time()
      for i, (lb1, lb2) in enumerate(pairs):
        console.print_progress(index=i, total=len(pairs))

        # (2.3.1) Check distance key
        if (lb1, lb2) in dist_dict and not overwrite: continue

        # (2.3.2) Fetch data dictionaries
        key_1, key_2 = (lb1, ck, pk), (lb2, ck, pk)
        data_1, data_2 = nebula.data_dict[key_1], nebula.data_dict[key_2]

        # (2.3.3) Calculate KDE distance
        try:
          d = hm.calc_distance(data_1, data_2, key_1, key_2,
                               conditional=conditional,
                               shift_compensation=compensate_shift)
          dist_dict[(lb1, lb2)] = d
        except: console.warning(
          f'Failed to calculate KDE distance between {key_1} and {key_2} !')

      # (2.4) Save buffer for each
      console.show_status(f'Time elapsed: {time.time() - tic:.3f}s')
      self.save_buffer()

  # endregion: - Distance Calculation

  # region: - Omix Generation

  def gen_KDE_omix(self, time_resolution, channels, probe_keys,
                   conditional=True, model_key='model_1',
                   compensate_shift=True) -> Omix:
    conditional = int(conditional)

    # (1) Confirm label pairs
    label_pairs, sample_names, targets = self.pairs_names_targets

    # (2)
    chn_prob_keys = [(ck, pk) for ck in channels for pk in probe_keys]
    features = np.zeros(shape=(len(targets), len(chn_prob_keys)))
    feature_names = []
    for i, (ck, pk) in enumerate(chn_prob_keys):
      buffer_key = self._get_buffer_key(
        model_key, conditional, time_resolution, ck, pk, compensate_shift)

      feature_name = self._get_feature_key(
        model_key, conditional, ck, pk, compensate_shift)
      feature_names.append(feature_name)

      for j, (lb1, lb2) in enumerate(label_pairs):
        features[j, i] = self.buffer[buffer_key][(lb1, lb2)]

    # (3) Wrap and return
    omix = Omix(features, targets, feature_labels=feature_names,
                sample_labels=sample_names,
                target_labels=['Not Match', 'Match'])

    return omix

  # endregion: - Omix Generation

  # region: - IO

  def save_buffer(self):
    io.save_file(self.buffer, self.buffer_path, verbose=True)

  # endregion: - IO

  # region: - Figures

  def plot_age_delta_distance(self, time_resolution, channels, probe_keys,
                              label_pairs, ad_dict, patient_dict: dict,
                              model_key='model_1', cond=1, comp=1,
                              figsize=(7, 5)):
    from matplotlib.lines import Line2D
    from pictor import Pictor

    import matplotlib.pyplot as plt

    probe_keys.remove('KURT')
    console.show_status('`KURT` removed from probe_keys.')

    category_colors = {'PSG Diagnostic': 'red', 'Master': 'blue',
                       'PSG': 'green', 'Other': 'gray'}
    def plot(fig: plt.Figure, x):
      pk = x
      assert len(channels) == 6

      # Create layout
      for ci, ck in enumerate(channels):
        ax: plt.Axes = fig.add_subplot(2, 3, ci + 1)
        ax.set_title(ck)

        X, Y, C = [], [], []
        for pair_key in label_pairs:
          X.append(ad_dict[pair_key])
          buffer_key = self._get_buffer_key(
            model_key, cond, time_resolution, ck, pk, comp)
          Y.append(self.buffer[buffer_key][pair_key])

          # Set category
          lb1, lb2 = pair_key
          pid, sid_1 = lb1.split('_')
          pid, sid_2 = lb2.split('_')
          st_1, st_2 = [patient_dict[pid][sid]['study_type']
                        for sid in (sid_1, sid_2)]
          if st_1 == st_2: C.append(category_colors[st_1])
          else: C.append(category_colors['Other'])

        # Plot scatter points
        ax.scatter(X, Y, c=C, alpha=0.5, s=3)
        # ax.scatter(X, Y, alpha=0.5, label=pk if ci == 0 else None)

        ax.set_xlabel('Years')
        ax.set_ylabel('KDE Distance')
        ax.set_ylim([0, 1])

      # Create custom legend elements
      # legend_elements = [
      #   Line2D([0], [0], marker='o', color='w',
      #          markerfacecolor=color, markersize=8, label=category)
      #   for category, color in category_colors.items()
      # ]

      # Add legend to the figure
      # fig.legend(handles=legend_elements, title='Study Types')

      fig.suptitle(f'[{x}] Cond={cond}, Comp={comp}')
      # fig.legend()
      fig.tight_layout()

    # Create pictor and show
    p = Pictor(f'{self.name}_{time_resolution}s', figure_size=figsize)
    p.add_plotter(plot)
    p.objects = probe_keys
    p.show()

  def plot_auc_rank(self, channels, probe_keys, time_resolution,
                    model_key='model_1', conds=(0, 1), comps=(0, 1),
                    figsize=(7, 5)):
    """Conclusion: """
    if conds in (0, 1): conds = (conds, )
    if comps in (0, 1): comps = (comps, )

    # (1) Generate AUC dict
    from pictor.xomics.evaluation.roc import ROC

    label_pairs, _, targets = self.pairs_names_targets

    # .. auc_dict.keys = [(ck, pk, cond, comp), ...]
    auc_dict = {}
    ck_pk_cond_comp_set = [(ck, pk, cond, comp)
                           for ck in channels for pk in probe_keys
                           for cond in conds for comp in comps]
    for ck, pk, cond, comp in ck_pk_cond_comp_set:
      # .. get buffer
      buffer_key = self._get_buffer_key(
        model_key, cond, time_resolution, ck, pk, comp)

      features = [-self.buffer[buffer_key][(lb1, lb2)]
                  for lb1, lb2 in label_pairs]
      auc_dict[(ck, pk, cond, comp)] = ROC(features, targets).auc

    # (2) Plot
    import matplotlib.pyplot as plt

    def probe_score(pk):
      return max([auc_dict[(ck, pk, cond, comp)]
                  for ck in channels for cond in conds for comp in comps])
    probe_keys = sorted(probe_keys, key=probe_score)

    # .. configuration
    delta = 0.15
    ms = 7
    # colors = ['#af4141', '#3b6ea9']
    colors = ['#6b96ca', '#c76a67', '#e19a52',
              '#c3c33d', '#79b051', '#74babb']

    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_subplot(1, 1, 1)
    for i, pk in enumerate(probe_keys):
      y = i

      for j, ck in enumerate(channels):
        # yj = y - (j - 0.5) * 2 * delta
        yj = y - 0.5 + 1 / (len(channels) + 1) * (j + 1)

        # .. Plot marker
        auc_dict_pk_ck = {(cond, comp): auc_dict[(ck, pk, cond, comp)]
                          for cond in conds for comp in comps}

        label = f'{ck}' if i == 0 else None
        color = colors[j]

        for (cond, comp), auc in auc_dict_pk_ck.items():
          face_color = color if comp else 'white'
          marker = 10 if cond else 11
          ax.plot(auc, yj, marker=marker, markeredgecolor=color,
                  markersize=ms, markerfacecolor=face_color)

        # .. Plot line
        min_auc = min(auc_dict_pk_ck.values())
        max_auc = max(auc_dict_pk_ck.values())

        ax.plot([min_auc, max_auc], [yj, yj], color=color, label=label)

    # Draw split lines
    x0, x1 = ax.get_xlim()
    for i, pk in enumerate(probe_keys):
      if i == 0: continue
      # Get current limits
      ax.plot([x0, x1], [i - 0.5, i - 0.5], color='grey', linestyle=':')

    auc_macro = 0.784
    ax.plot([auc_macro, auc_macro], [-0.5, 10.5], linestyle='--',
            color='green', zorder=-99, alpha=0.5)
    ax.set_ylim([-0.5, len(probe_keys) - 0.5])

    # Rename probe keys
    KEY_MAP = {
      'FREQ-20': 'Frequency',
      'PR-BETA_TOTAL': 'Beta/Total',
      'PR-ALPHA_TOTAL': 'Alpha/Total',
      'PR-THETA_ALPHA': 'Theta/Alpha',
      'PR-THETA_TOTAL': 'Theta/Total',
      'KURT': 'Kurtosis',
      'PR-DELTA_TOTAL': 'Delta/Total',
      'AMP-1': 'Amplitude',
      'PR-DELTA_ALPHA': 'Delta/Alpha',
      'PR-DELTA_THETA': 'Delta/Theta',
      'POWER-30': 'Total Power',
    }
    probe_keys = [KEY_MAP[pk] for pk in probe_keys]

    # Set legends
    # ax.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
    for cond in conds:
      marker = 10 if cond else 11
      for comp in comps:
        label = 'Stage-wise' if cond else 'Mixed'
        label += ' w SC' if comp else ' w/o SC'
        face_color = 'gray' if comp else 'white'
        ax.plot(-1, 0, marker=marker, markersize=ms, color='gray',
                markerfacecolor=face_color, label=label)

    ax.legend()

    ax.set_xlabel('Authentication AUC')
    ax.set_yticks(list(range(len(probe_keys))), probe_keys)
    ax.set_xlim([x0, x1])

    plt.tight_layout()
    plt.show()

  # endregion: - Figures

  # endregion: Public Methods

  # region: Private Methods

  def _get_feature_key(self, model_key, conditional, ck, pk, compensate_shift):
    cond_str = 'C' if conditional else 'NC'
    comp_str = 'CS' if compensate_shift else 'NCS'
    return f'{model_key}_{cond_str}_{ck}_{pk}_{comp_str}'

  def _get_buffer_key(self, model_key, conditional, time_resolution, ck, pk,
                      compensate_shift):
    key = [model_key, conditional, time_resolution, ck, pk]
    if not compensate_shift: key.append('wo_shift_compensation')
    return tuple(key)

  def _finalize_init(self):
    # Sanity check
    assert os.path.exists(self.work_dir), f'Path `{self.work_dir}` not found.'
    assert os.path.exists(self.cloud_dir), f'Path `{self.cloud_dir}` not found.'

    console.show_status('DistanceAgent initialized.', prompt='[DA]')
    console.supplement(f'Work dir: {self.work_dir}', level=2)
    console.supplement(f'Cloud dir: {self.cloud_dir}', level=2)

    if any([self.nights_1 is None, self.nights_2 is None]):
      console.show_status('Nights lists not provided', prompt='[DA]')
    else:
      assert len(self.nights_1) == len(self.nights_2)
      console.supplement(f'Subject # = {len(self.nights_1)}', level=2)

  # endregion: Private Methods
