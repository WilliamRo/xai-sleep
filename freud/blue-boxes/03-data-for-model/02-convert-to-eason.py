from freud.talos_utils.sleep_sets.sleepeason import SleepEason



# Set directories
data_dir = r'../../../data/'

src_folder = ['sleepedfx', 'ucddb', 'rrsh-night'][0]

src_dir = data_dir + src_folder
tgt_dir = data_dir + 'sleepeason1'

# Set src_pattern
src_pattern = '*.sg'

file_prefix = src_folder + '-'
if 'sleepedfx' in src_folder:
  src_pattern = '*(trim1800;iqr,1,20;128).sg'
elif 'ucddb' in src_folder:
  src_pattern = '*(iqr,1,20;128).sg'
  file_prefix = ''
elif 'rrsh' in src_folder:
  src_pattern = '*(trim;iqr,1,20).sg'
  file_prefix = 'rrsh-'

# Convert to signal groups
SleepEason.convert_to_eason_sg(src_dir, tgt_dir, src_pattern=src_pattern,
                               file_prefix=file_prefix)