import pingouin as pg
import pandas as pd



values_1 = [6, 3, 4, 5, 6.1, 2.9, 4.2, 4.8]
values_2 = [6, 6, 6, 6, 6.1, 5.9, 6.2, 5.8]
values_3 = [6.01, 6, 6, 6, 6., 6.01, 6.01, 6.01]
data = pd.DataFrame(
  data={'Subject': [1, 2, 3, 4, 1, 2, 3, 4],
        'Measurement': [1, 1, 1, 1, 2, 2, 2, 2],
        'Value': values_3}).round(3)
icc = pg.intraclass_corr(
  data, targets='Subject', raters='Measurement', ratings='Value')
print(icc)