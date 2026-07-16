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

ROOT = Path(__file__).resolve().parent.parent # project root
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

##########################
# SETUP AND DATA LOADING #
##########################

tmin, tmax = -1.0, 4.0 # seconds relative to each event onset
subjects = [1]         # subject
runs = [6, 10, 14]     # selected recordings

raw_fnames = eegbci.load_data(subjects, runs) # loads files, returns paths
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames]) # [RawEDF(run6), RawEDF(run10), RawEDF(run14)]
filtered = raw.copy()

###############################
# PREPROCESSING AND FILTERING #
###############################

eegbci.standardize(filtered) # standardizing channel names
montage = make_standard_montage("standard_1005")         # loading standard 1005
filtered.set_montage(montage)                            # setting raw to 1005
filtered.annotations.rename(dict(T1="hands", T2="feet")) # renaming
filtered.set_eeg_reference(projection=True)              # sets an average reference

filtered.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge") # filtering to 7-30 hz

# fig_raw = raw.compute_psd(fmax=60).plot()
# fig_filtered = filtered.compute_psd(fmax=60).plot()
# fig_raw.savefig(FIG_DIR/"raw.png")
# fig_filtered.savefig(FIG_DIR/"filtered.png")

###########################
# EPOCHING (SLICING DATA) #
###########################

# picking only eeg, ignoring stimulus channels, EOG (eye movement) channels and bad channels
picks = pick_types(filtered.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

epochs = Epochs(
    filtered,
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

# monte-carlo cross-validation
scores = []
epochs_data = epochs.get_data(copy=False)
epochs_data_train = epochs_train.get_data(copy=False)
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

#################################
# THE MACHINE LEARNING PIPELINE #
#################################

# classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

clf = Pipeline([("CSP", csp), ("LDA", lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1.0 - class_balance)
print(f"Classification accuracy: {np.mean(scores)} / Chance level: {class_balance}")

csp.fit_transform(epochs_data, labels)
spf = get_spatial_filter_from_estimator(csp, info=epochs.info)
spf.plot_scree()
spf.plot_patterns(components=np.arange(4))
plt.show()
