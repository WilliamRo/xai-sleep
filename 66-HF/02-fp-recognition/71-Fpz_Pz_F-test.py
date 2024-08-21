from hypnomics.freud.freud import Freud
from hypnomics.freud.nebula import Nebula
from hypnomics.hypnoprints.stat_models.model_1 import HypnoModel1
from hf.sc_tools import get_dual_nebula
from hf.sc_tools import CK_MAP, PK_MAP
from roma import console
from roma import finder
from x_dual_view import PAIRED_LABELS

import os



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
WORK_DIR = r'../data/sleepedfx_sc'
CHANNELS = [
  'EEG Fpz-Cz',
  'EEG Pz-Oz'
]

PROBE_KEYS = [
  'FREQ-20',   # 0
  'RP-BETA',
]

# SG_LABELS = ['SC4001E', 'SC4002E']
SG_LABELS = finder.walk(WORK_DIR, type_filter='dir', return_basename=True)[:999]

# [ 2, 5, 10, 30, ]
TIME_RESOLUTION = 30

NEB_FN = f'SC-{TIME_RESOLUTION}-KDE-0730.nebula'
neb_file_path = os.path.join(WORK_DIR, NEB_FN)

ck1 = CHANNELS[0]
ck2 = CHANNELS[0]
pk1 = PROBE_KEYS[0]
pk2 = PROBE_KEYS[1]

sample_id = 0
compare_id = 1
# -----------------------------------------------------------------------------
# (2) Load paired nebula
# -----------------------------------------------------------------------------
assert os.path.exists(neb_file_path)
nebula: Nebula = Nebula.load(neb_file_path)

nebula.set_labels(PAIRED_LABELS)
neb_1, neb_2 = get_dual_nebula(nebula)

l1, l2 = neb_1.labels[sample_id], neb_2.labels[sample_id]
lc = neb_2.labels[compare_id]

ck1pk1 = f'{CK_MAP[ck1]}-{PK_MAP[pk1]}'
ck2pk2 = f'{CK_MAP[ck2]}-{PK_MAP[pk2]}'

key_1_1, key_1_2 = (l1, ck1, pk1), (l1, ck2, pk2)
key_2_1, key_2_2 = (l2, ck1, pk1), (l2, ck2, pk2)
key_c_1, key_c_2 = (lc, ck1, pk1), (lc, ck2, pk2)
# -----------------------------------------------------------------------------
# (3) Run test
# -----------------------------------------------------------------------------
console.show_info('Testing details:')
console.supplement(f'Samples: ({l1}, {l2}), {lc}', level=2)
console.supplement(f'Channel-Probe: ({ck1}x{pk1}), ({ck2}x{pk2})', level=2)

# (3.1) Fetch data
data_1_pk1 = neb_1.data_dict[(l1, ck1, pk1)]
data_1_pk2 = neb_1.data_dict[(l1, ck2, pk2)]
data_2_pk1 = neb_2.data_dict[(l2, ck1, pk1)]
data_2_pk2 = neb_2.data_dict[(l2, ck2, pk2)]
data_c_pk1 = neb_2.data_dict[(lc, ck1, pk1)]
data_c_pk2 = neb_2.data_dict[(lc, ck2, pk2)]

hm = HypnoModel1()
# (3.2) Report distance
d_same_pk1 = hm.calc_distance(data_1_pk1, data_2_pk1, key_1_1, key_2_1)
d_same_pk2 = hm.calc_distance(data_1_pk2, data_2_pk2, key_1_2, key_2_2)
d_compare_pk1 = hm.calc_distance(data_1_pk1, data_c_pk1, key_1_1, key_c_1)
d_compare_pk2 = hm.calc_distance(data_1_pk2, data_c_pk2, key_1_2, key_c_2)

console.show_info('1-D Distance')
console.supplement(f'D({l1},{l2})@({ck1pk1}) = {d_same_pk1:.3f}', level=2)
console.supplement(f'D({l1},{l2})@({ck2pk2}) = {d_same_pk2:.3f}', level=2)
console.supplement(f'D({l1},{lc})@({ck1pk1}) = {d_compare_pk1:.3f}', level=2)
console.supplement(f'D({l1},{lc})@({ck2pk2}) = {d_compare_pk2:.3f}', level=2)

# (3.3) Report joint distance
d_same_pk1_pk2 = hm.calc_joint_distance(
  (data_1_pk1, data_1_pk2), (data_2_pk1, data_2_pk2),
  (key_1_1, key_1_2), (key_2_1, key_2_2))
d_compare_pk1_pk2 = hm.calc_joint_distance(
  (data_1_pk1, data_1_pk2), (data_c_pk1, data_c_pk2),
  (key_1_1, key_1_2), (key_c_1, key_c_2))

console.show_info('2-D Distance')
console.supplement(f'D({l1},{l2})@({ck1pk1}x{ck2pk2}) = {d_same_pk1_pk2:.3f}',
                   level=2)
console.supplement(f'D({l1},{lc})@({ck1pk1}x{ck2pk2}) = {d_compare_pk1_pk2:.3f}',
                   level=2)

