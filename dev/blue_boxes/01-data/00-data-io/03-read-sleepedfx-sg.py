from freud.talos_utils.sleep_sets.sleepedfx import SleepEDFx



data_root = r'../../../../data/sleepedf'

ds = SleepEDFx.load_as_sleep_set(data_root)
ds.show()

