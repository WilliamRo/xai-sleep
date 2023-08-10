from freud.talos_utils.sleep_sets.sleepeason import SleepEason



# src_dir = r'../../../data/sleepedfx'
# src_dir = r'../../../data/ucddb'
src_dir = r'../../../data/rrsh'
# src_dir = r'../../../data/sleepedfx'
tgt_dir = r'../../../data/sleepeason'

SleepEason.convert_to_eason_sg(src_dir, tgt_dir, src_pattern='*.sg')