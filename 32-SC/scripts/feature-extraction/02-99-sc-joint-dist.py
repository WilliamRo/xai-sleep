from ew_viewer import EWViewer
from tframe.utils.file_tools.io_utils import load



FP_PATH = r'../../features/0211-N153-ch(all)-F(20)-A(128).fp'
EXPORT_PATH = r'../../features/sub-data-02'


results = load(FP_PATH)
meta = results['meta']

ew = EWViewer(walker_results=results)
ew.show()

"""
commands: fpa
"""
