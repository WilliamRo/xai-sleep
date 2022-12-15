from freud.talos_utils.sleep_sets.ucddb import UCDDB



data_root = r'../../../../../data/ucddb'

ds = UCDDB.load_as_sleep_set(data_root)
ds.show()

