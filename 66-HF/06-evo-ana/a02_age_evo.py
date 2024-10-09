from a01_age_vis import nebula, PK1, PK2, Telescope
from roma import console



# -----------------------------------------------------------------------------
# (1) Configuration
# -----------------------------------------------------------------------------
intervals = [
  (low, low + 10)
  for low in range(20, 100, 5)
]

# Report distribution
console.show_info(f'Distribution of age (totally {len(nebula.labels)}):')
for l, h in intervals:
  N = len([k for k, v in nebula.meta.items() if l <= v['age'] < h])
  console.supplement(f'{l} <= age < {h}: {N} samples', level=2)

# exit()
# -----------------------------------------------------------------------------
# (2) Generate evolution
# -----------------------------------------------------------------------------
evo = nebula.gen_evolution('age', intervals)

# -----------------------------------------------------------------------------
# (3) Visualization
# -----------------------------------------------------------------------------
if __name__ == '__main__':
  configs = {
    # 'xmin': 3, 'xmax': 9, 'ymin': 0, 'ymax': 200,
    'show_kde': 0,
    'show_scatter': 1,
    'show_vector': 0,
    # 'scatter_alpha': 0.05,
  }

  viewer_class = Telescope
  viewer_configs = {'plotters': 'HA'}
  evo.dual_view(x_key=PK1, y_key=PK2, viewer_class=viewer_class,
                viewer_configs=viewer_configs, **configs)
