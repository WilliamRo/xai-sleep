import os
from roma import finder



src_path = os.path.abspath('.')
dst_path = r'\\192.168.5.100\xai-beta\xai-sleep'

# Comment the folder name you want to synchronize
ignored_patterns=('.*', '__*__', 'checkpoints', 'logs', 'tests',
                  '00-XSLP',
                  '01-DSN',
                  '02-USLP',
                  '03-SENET',
                  '06-ATTN',
                  '08-FNN',
                  '09-S2S',
                  '10-LEG',
                  '11-RNN',
                  '20-CAM',
                  '21-HO',
                  '30-FRE',
                  '33-ST',
                  '50-NARCO',
                  '51-MASS',
                  'xai-kit',
                  # 'freud',
                  # 'hypnomics',
                  )

FOLDER_NAME = [None,
               '26-HSP', # 1
               # '31-OSA-XU', # 2
               '32-SC', # 2
               r'hypnomics\hypnomics\freud', # 3
               'freud', # 4
               '66-HF', # 5
               r'hypnomics\hypnomics', # 6
               ][4]

if FOLDER_NAME is not None:
  src_path = os.path.join(src_path, FOLDER_NAME)
  dst_path = os.path.join(dst_path, FOLDER_NAME)

finder.synchronize(src_path, dst_path, pattern='*.py',
                   ignored_patterns=ignored_patterns, verbose=True)
