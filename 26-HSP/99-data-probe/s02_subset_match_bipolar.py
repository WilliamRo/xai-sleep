"""
Last modified: 2024-12-26
"""
# Add path in order to be compatible with Linux
import sys, os

SOLUTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

PATH_LIST = ['26-HSP', '26-HSP/99-data-probe', 'xai-kit', 'hypnomics',
             'xai-kit/roma', 'xai-kit/pictor', 'xai-kit/tframe']

if __name__ == '__main__':
  print(f'Solution dir = {SOLUTION_DIR}')
  sys.path.append(SOLUTION_DIR)
  for p in PATH_LIST: sys.path.append(os.path.join(SOLUTION_DIR, p))

from roma import io

import a00_common as hub



# -----------------------------------------------------------------------------
# (0) Configuration
# -----------------------------------------------------------------------------
SRC_SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_378
TGT_SUBSET_FN = hub.SubsetDicts.ss_2ses_3types_bipolar_218
CHANNELS = ('Fpz', 'Cz', 'Pz', 'Oz')

# -----------------------------------------------------------------------------
# (1) Load and filter subset
# -----------------------------------------------------------------------------
sub_dict = hub.ha.load_subset_dict(SRC_SUBSET_FN)
sub_sub_dict = hub.ha.filter_patients_by_channels(
  sub_dict, CHANNELS, min_n_sessions=2, verbose=True)

# -----------------------------------------------------------------------------
# (2) Save subset
# -----------------------------------------------------------------------------
io.save_file(sub_sub_dict, os.path.join(hub.META_DIR, TGT_SUBSET_FN),
             verbose=True)


# # Preload
# console.show_status('Pre-loading ...')
# for i, ho in enumerate(subset_hos):
#   console.print_progress(i, len(subset_hos))
#   _ = ho.channel_dict
#
# for ck in ('F3-M2', 'Fpz', 'Cz', 'Pz', 'Oz', 'Fpz-Cz'):
#   n = len([ho for ho in subset_hos if ck in ho.channel_dict])
#   console.show_status(f'{n} PSGs have channel `{ck}`.')
#

