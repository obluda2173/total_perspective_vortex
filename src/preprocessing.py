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

path = "../data/files/" # not used right now

tmin, tmax = -1.0, 4.0 # seconds relative to each event onset
subjects = [1]         # subject
runs = [6, 10, 14]     # selected recordings

raw_fnames = eegbci.load_data(subjects, runs) # loads files, returns paths

raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) # [RawEDF(run6), RawEDF(run10), RawEDF(run14)]

eegbci.standardize(raw) # standardizing channel names

montage = make_standard_montage("standard_1005")    # loading standard 1005
raw.set_montage(montage)                            # setting raw to 1005
raw.annotations.rename(dict(T1="hands", T2="feet")) # renaming
raw.set_eeg_reference(projection=True)              # sets an average reference

raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge") # filtering to 7-30 hz

# picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads") # not sure
picks = ["C3", "C4", "Cz"]

epochs = Epochs(
    raw,
    event_id=["hands", "feet"],
    tmin=tmin,
    tmax=tmax,
    proj=True,
    picks=picks,
    baseline=None,
    preload=True,
) # slices of raw, one per event

epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0) # cropped copy of epochs
labels = epochs.events[:, -1] - 2

epochs_train.plot(scalings='auto')
plt.show()
