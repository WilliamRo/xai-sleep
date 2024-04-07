"""
This script has been capisulated into sc.sc_agent
"""
from roma import console

import pandas as pd



# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
xls_path = r'../../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
SHOW_STATISTICAL_CHARTS = True

# ----------------------------------------------------------------------------
# Read data
# ----------------------------------------------------------------------------
df = pd.read_excel(xls_path)
all_subjects = df['subject'].unique()
valid_subjects = [
  s for s in all_subjects if len(df[df['subject'] == s]) > 1]
invalid_subjects = sorted(list(set(all_subjects) - set(valid_subjects)))

# ----------------------------------------------------------------------------
# Report data details
# ----------------------------------------------------------------------------
console.show_status('SC-subjects information:')
console.supplement(f'Total subject #: {len(all_subjects)}', level=2)
console.supplement(f'Valid subject #: {len(valid_subjects)}', level=2)
console.supplement(f'Invalid subject IDs: {invalid_subjects}', level=2)

# ----------------------------------------------------------------------------
# Filter data
# ----------------------------------------------------------------------------
pinfo = df[['subject', 'age', 'sex (F=1)']]
pinfo = pinfo.rename(columns={'sex (F=1)': 'sex'})
pinfo = pinfo[pinfo['subject'].isin(valid_subjects)]
pinfo = pinfo.drop_duplicates()
pinfo.loc[pinfo['sex'] == 1,'sex'] = 'F'
pinfo.loc[pinfo['sex'] == 2,'sex'] = 'M'

# ----------------------------------------------------------------------------
# Show statistical charts if required
# ----------------------------------------------------------------------------
if SHOW_STATISTICAL_CHARTS:
  import matplotlib.pyplot as plt
  import seaborn as sns

  sns.set_theme(rc={'figure.figsize': (5, 6)})

  sns.violinplot(data=pinfo, y='age', hue='sex', split=True, fill=False,
                 inner='quart', palette='muted')

  plt.tight_layout()
  plt.show()



