from freud.talos_utils.sleep_sets.rrshv1 import RRSHSCv1



data_root = r'../../../../../data/rrsh'

ds = RRSHSCv1.load_as_sleep_set(data_root)
ds.show()

