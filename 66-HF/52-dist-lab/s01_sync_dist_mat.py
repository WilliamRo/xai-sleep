from roma import console
from roma import io



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
SRC_PATH = r"P:\xai-sleep\data\sleepedfx-sc\sc_dist\SC10.da"
TGT_PATH = r"P:\xai-sleep\data\sleepedfx-sc\sc_dist\SC75.da"

# -----------------------------------------------------------------------------
# (2) Read and synchronize
# -----------------------------------------------------------------------------
src_dict: dict = io.load_file(SRC_PATH, verbose=True)
tgt_dict: dict = io.load_file(TGT_PATH, verbose=True)

for key, dist_dict in src_dict.items():
  if key not in tgt_dict:
    tgt_dict[key] = dist_dict
    console.show_status(f'Set {key} to target.')
    continue

  for pair_key, dist_val in dist_dict.items():
    n = 0
    if pair_key not in tgt_dict[key]:
      tgt_dict[key][pair_key] = dist_val
      n += 1
    if n > 0:
      console.show_status(f'Set {n} new distances for {key}.')

io.save_file(tgt_dict, TGT_PATH, verbose=True)

