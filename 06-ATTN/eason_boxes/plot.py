import sys, os
import numpy as np
from pictor import Pictor
from tframe.utils.note import Note
import matplotlib.pyplot as plt

class Plot():
  def __init__(self, data, xlabel=None, ylabel=None, label='origin', tilte=None):
    """
    param: data {x1:[value1, value2,...], x2:[value1, value2,...]}

    to plot [(x1, x2, ...) (fun(values), ... )]
    """
    self.x = []
    self.mean_all = []
    self.confidence_high_all = []
    self.confidence_low_all = []

    self.data = data
    self.xlabel = xlabel
    self.ylabel = ylabel
    self.label = label
    self.title = tilte
    # self.y = y

  def cal_data_from_dic(self, dic):

  # calculate statistical parameter in per ratio
    for key, value_list in dic.items():
      mean = np.mean(value_list)
      std = np.std(value_list)
      confidence_high = mean + 1.96 * std / np.sqrt(len(value_list))
      confidence_low = mean - 1.96 * std / np.sqrt(len(value_list))

      self.mean_all.append(mean)
      self.confidence_high_all.append(confidence_high)
      self.confidence_low_all.append(confidence_low)
      self.x.append(key)

  def plotter(self, ax: plt.Axes):

    ax.plot(self.x, self.mean_all, color='red', linewidth=1, alpha=1,
            marker='s', label=self.label)
    ax.fill_between(self.x, self.confidence_low_all, self.confidence_high_all,
                    alpha=0.3, color='red')

    ax.set_xlabel(self.xlabel, loc='right')
    ax.set_ylabel(self.ylabel, loc='top')
    ax.set_title(self.title, fontweight='bold', loc='center')
    ax.legend()
  def show(self):
    p = Pictor(figure_size=(8, 5))
    self.cal_data_from_dic(self.data)
    plotter = p.add_plotter(self.plotter)
    p.show()
