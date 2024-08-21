from pictor.xomics.omix import Omix
from pictor.xomics.evaluation.pipeline import Pipeline



# -----------------------------------------------------------------------------
# (1) Read Omix
# -----------------------------------------------------------------------------
file_path = [
  r'./data/0812_age_01.omix',
][0]
omix = Omix.load(file_path)

# omix.show_in_explorer()
# exit()
# -----------------------------------------------------------------------------
# (2) Feature selection
# -----------------------------------------------------------------------------
pi = Pipeline(omix, ignore_warnings=1, save_models=1)

pi.report()
# pi.plot_matrix()

pkg = pi.pipeline_ranking[0][-1]
pkg.report(ra=1)
