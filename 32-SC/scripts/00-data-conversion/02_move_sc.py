from roma import finder

import os
import shutil



src_dir = r'P:\xai-sleep\data\sleep-edf-database-expanded-1.0.0\sleep-cassette'
tgt_dir = r'P:\xai-sleep\data\sleepedfx-sc\sc_sg'

# file_pattern = '*(trim1800;iqr,1,20;128).sg'
file_pattern = f'*(trim1800;128).sg'
file_list = finder.walk(src_dir, pattern=file_pattern)

for src_file in file_list:
  # tgt_file = os.path.join(tgt_dir, os.path.basename(src_file))
  shutil.move(src_file, tgt_dir)



