from pictor.objects.signals.signal_group import SignalGroup, Annotation
from freud.talos_utils.slp_config import SleepConfig
from freud.talos_utils.slp_set import SleepSet

import importlib.util



# (0) Load module from t-file
t_file_path = r''
module_name = 'this_name_does_not_matter'
spec = importlib.util.spec_from_file_location(module_name, t_file_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

th: SleepConfig = mod.core.th
th.developer_code += 'deactivate'

# Execute main to load basic module settings
mod.main(None)
th.use_batch_mask = False

# (1) Prepare data set
# Get displayed signal_group TODO
sg: SignalGroup = None
ds = SleepSet(signal_groups=[sg])

# Set CHANNELS for extracting tapes during configuration
ds.CHANNELS = {f'{i}': k for i, k in enumerate(self.channel_list)}
th.data_config = f'whatever {channels}'
ds.configure()

ds = ds.extract_data_set(include_targets=False)

# (2) Prepare model
from tframe import Classifier, context

# Do some cleaning
context.logits_tensor_dict = {}

if model_name is None: model_name = mod.model_name
model: Classifier = th.model()

stft_tensor = model.children[3].output_tensor
results = model.evaluate(fetches=stft_tensor, data=ds)

batch_size = 1 if th.use_rnn else 128
preds = model.classify(ds, batch_size=batch_size, verbose=True)
model.shutdown()

# (3) Visualize