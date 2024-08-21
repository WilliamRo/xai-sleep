import matplotlib.pyplot as plt
import numpy as np



def calc_mae(ages, est_ages):
  d = {}
  for age, est_age in zip(ages, est_ages):
    k = age // 5
    if k not in d: d[k] = []
    d[k].append(abs(est_age - age))
  mea_list = [np.average(v) for v in d.values()]
  return np.average(mea_list), np.std(mea_list)


def plot_age_est(age, est_age, train_data=None):
  # Plot ideal line
  min_age, max_age = min(age), max(age)
  plt.plot([min_age, max_age], [min_age, max_age], 'r--')

  # Plot train data if provided
  if train_data is not None:
    train_age, train_est_age = train_data
    mu, sigma = calc_mae(train_age, train_est_age)
    legend = f'Train MAE = {mu:.2f} ± {sigma:.2f} years'
    plt.plot(train_age, train_est_age, 'o', color='orange',
             label=legend, alpha=0.5)

  # Plot test data
  mu, sigma = calc_mae(age, est_age)
  legend = f'Test MAE = {mu:.2f} ± {sigma:.2f} years'
  plt.plot(age, est_age, 'o', label=legend)

  # Plot settings
  plt.xlabel('Age')
  plt.ylabel('Estimated Age')

  plt.legend()

  plt.tight_layout()
  plt.show()
