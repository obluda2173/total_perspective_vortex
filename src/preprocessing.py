import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP, get_spatial_filter_from_estimator
from mne.io import concatenate_raws, read_raw_edf

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project root
DATA_DIR = "../data/files/" # not used right now
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

tmin, tmax = -1.0, 4.0 # seconds relative to each event onset
subjects = [1]         # subject
runs = [6, 10, 14]     # selected recordings

raw_fnames = eegbci.load_data(subjects, runs) # loads files, returns paths
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) # [RawEDF(run6), RawEDF(run10), RawEDF(run14)]
filtered = raw.copy()

eegbci.standardize(filtered) # standardizing channel names

montage = make_standard_montage("standard_1005")         # loading standard 1005
filtered.set_montage(montage)                            # setting raw to 1005
filtered.annotations.rename(dict(T1="hands", T2="feet")) # renaming
filtered.set_eeg_reference(projection=True)              # sets an average reference

filtered.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge") # filtering to 7-30 hz

fig_raw = raw.compute_psd(fmax=60).plot()
fig_filtered = filtered.compute_psd(fmax=60).plot()
fig_raw.savefig(FIG_DIR/"raw.png")
fig_filtered.savefig(FIG_DIR/"filtered.png")

# picks = pick_types(raw_filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads") # not sure

# epochs = Epochs(
#     raw_filtered,
#     event_id=["hands", "feet"],
#     tmin=tmin,
#     tmax=tmax,
#     proj=True,
#     picks=picks,
#     baseline=None,
#     preload=True,
# ) # slices of raw, one per event

# epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0) # cropped copy of epochs
# labels = epochs.events[:, -1] - 2

# epochs_train.plot(scalings='auto')
# plt.show()
