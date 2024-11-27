import os
from roma import finder
from roma import console


src_path = os.path.abspath('.')
dst_path = r'T:\william\xai-alfa'

ignored_patterns=('.*', '__*__', 'checkpoints', 'logs', 'tests')
finder.synchronize(src_path, dst_path, pattern='*.py',
                   ignored_patterns=ignored_patterns, verbose=True)