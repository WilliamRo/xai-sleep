from hypnomics.hypnoprints.hp_extractor import extract_hypnoprints_from_hypnocloud
from pictor.xomics.omix import Omix
from collections import OrderedDict
from roma import io, console

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



XLSX_PATH = r'../../../data/sleep-edf-database-expanded-1.0.0/SC-subjects.xls'
df = pd.read_excel(XLSX_PATH)


male_ages, female_ages = [], []
pids = sorted(list(set(df['subject'])))
for pid in pids:
  age = df.loc[df['subject'] == pid, 'age'].values[0]
  gender = df.loc[df['subject'] == pid, 'sex (F=1)'].values[0]
  if gender == 1: female_ages.append(age)
  else:
    assert gender == 2
    male_ages.append(age)

# Plot age distribution
# alpha = 0.5
# bins = np.linspace(10, 110, 10)
# plt.hist(male_ages, bins, label='Male', alpha=alpha)
# plt.hist(female_ages, bins, label='Female', alpha=alpha)
# plt.legend()

print(f'Male ({len(male_ages)}): [{np.min(male_ages)}, {np.max(male_ages)}]')
print(f'Female ({len(female_ages)}): [{np.min(female_ages)}, {np.max(female_ages)}]')

plt.figure(figsize=(7, 2))

v1 = plt.violinplot([male_ages], showmeans=True, vert=False, side='high')
v2 = plt.violinplot([female_ages], showmeans=True, vert=False, side='low')
plt.legend([v1['bodies'][0], v2['bodies'][0]], ['Male', 'Female'],
           loc='upper right')
# ticks = ['Male', 'Female']
plt.yticks([])
plt.xlabel('Age')

plt.tight_layout()
plt.show()


