from pictor.xomics.omix import Omix



# -----------------------------------------------------------------------------
# Load data (Do not modify codes in this section)
# -----------------------------------------------------------------------------
data_path = r'../data/radiomics-111x851.omix'

omix = Omix.load(data_path)
omix.report()

X = omix.features
y = omix.targets
# -----------------------------------------------------------------------------
# Import packages and construct machine learning workflow
# -----------------------------------------------------------------------------
