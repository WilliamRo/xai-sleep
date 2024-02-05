import pandas as pd

from roma import finder



# Get all indices in original excel file
excel_path = r'../../../data/rrsh-osa/OSA-wm.xlsx'
df = pd.read_excel(excel_path)
all_indices = list(df['序号'])

# Get valid sg files
data_dir = r'F:\20240203-osa-sg'
pattern = f'*(trim;simple;100).sg'
sg_file_list = finder.walk(data_dir, pattern=pattern)
valid_ids = sorted([int(fn.split('/')[2].split('(')[0])
                    for fn in sg_file_list])

# Sanity check
for i in valid_ids: assert i in all_indices
for i in all_indices:
  if i not in valid_ids: print(i)
