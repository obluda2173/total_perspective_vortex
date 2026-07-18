"""
Verification harness: your CSP vs mne.decoding.CSP.

This file deliberately implements NONE of CSP. It treats both estimators as
black boxes, reads their fitted `.filters_` and `.evals_`, and compares them.
Building covariances / solving the eigenproblem / choosing the ordering is your
job; none of it lives here.

Your estimator must expose, after `fit(X, y)`:
    - self.filters_ : ndarray (n_ch, n_ch), rows are spatial filters
    - self.evals_   : ndarray (n_ch,), the per-filter generalized eigenvalue,
                      row-aligned with filters_  (mirror MNE's attribute names)
and be a scikit-learn transformer (BaseEstimator + TransformerMixin) so it can
drop into the same Pipeline. If you name things differently, adjust the two
getters below and nothing else.

Run:  pytest -q test_csp.py
"""

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne.datasets import eegbci
from mne.decoding import CSP
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne import Epochs, pick_types

from csp import MyCSP

# ---- your implementation -----------------------------------------------------
# from mycsp import MyCSP           # <-- your CSP transformer
# MyCSP = CSP
# MyCSP = csp                         # placeholder so this file runs before yours
# ------------------------------------------------------------------------------

N_COMPONENTS = 4
FILTER_RTOL = 1e-6      # tighten to 1e-9 once you're confident
EVAL_ATOL = 1e-6
ACC_FLOOR_PTS = 0.01    # end-to-end accuracy must land within 1 point


# ============================================================ fixtures ========

@pytest.fixture(scope="module")
def data():
    """Single-subject, single-window array — the frozen baseline oracle.

    ONE array feeds both CSPs. No crop-vs-full-window mismatch, no CV leakage
    into the fit used for filter comparison.
    """
    subjects, runs = [1], [6, 10, 14]
    fnames = eegbci.load_data(subjects, runs)
    raw = concatenate_raws([read_raw_edf(f, preload=True) for f in fnames])

    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"))
    raw.annotations.rename(dict(T1="hands", T2="feet"))
    # raw.set_eeg_reference(projection=True)
    raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

    picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                       eog=False, exclude="bads")
    epochs = Epochs(raw, event_id=["hands", "feet"], tmin=-1.0, tmax=4.0,
                    proj=True, picks=picks, baseline=None, preload=True)
    epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)

    X = epochs_train.get_data(copy=False)
    # NOTE: explicit event_id map, not the `- 2` magic offset. Both CSPs see the
    # same y, so the offset can't cause a divergence — but wrong labels would
    # silently invalidate the *meaning* of the whole check, so make it explicit.
    y = epochs_train.events[:, -1]
    y = np.where(y == epochs.event_id["hands"], 0, 1)
    return X, y


@pytest.fixture(scope="module")
def fitted(data):
    X, y = data
    mne_csp = CSP(n_components=N_COMPONENTS, reg=None, log=True,
                  norm_trace=False).fit(X, y)
    mine = MyCSP(n_components=N_COMPONENTS, reg=None, log=True,
                 norm_trace=False).fit(X, y)
    return mne_csp, mine, X, y


# ============================================================ helpers ==========

def _unit_rows(W):
    """Kill the per-filter scale ambiguity: log-variance features are invariant
    to filter scaling, so W is only defined up to a per-row scalar."""
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def _match_rows(A, B):
    """Pair rows of A to rows of B ignoring order AND sign.

    Order: CSP components have no canonical position (mutual_info vs eigenvalue
    ordering differ), so match by similarity, not by index.
    Sign:  w and -w give identical variance, so compare |cos|.
    Returns (row_index_in_A, row_index_in_B, signed_cosine) per matched pair.
    """
    A, B = _unit_rows(A), _unit_rows(B)
    cos = A @ B.T                     # (n, n) cosine similarities, rows unit-norm
    ai, bj = linear_sum_assignment(-np.abs(cos))   # maximize |cos|, 1-to-1
    return ai, bj, cos[ai, bj]


# ============================================================ tests ============

def test_eigenvalue_spectrum(fitted):
    """Cleanest scalar check: sorted eigenvalues must agree.

    Eigenvalues have no sign/scale ambiguity; sorting removes the order one.
    This alone catches most covariance-construction mistakes (e.g. per-epoch
    averaging vs concat) because they move the spectrum.
    """
    mne_csp, mine, _, _ = fitted
    a = np.sort(mne_csp.evals_)
    b = np.sort(mine.evals_)
    assert np.allclose(a, b, atol=EVAL_ATOL), \
        f"eigenvalue spectra differ:\n mne ={a}\n mine={b}\n diff={a - b}"


def test_eigenvalue_invariants(fitted):
    """Self-consistency of your spectrum (from the paper, Eq. 4 normalization):
    every eigenvalue in [0, 1]. Cheap smoke test that runs even without MNE
    parity."""
    _, mine, _, _ = fitted
    e = mine.evals_
    assert np.all(e >= -EVAL_ATOL) and np.all(e <= 1 + EVAL_ATOL), \
        f"eigenvalues outside [0,1]: {np.sort(e)}"


def test_filters_match(fitted):
    """The tight test. After canceling order/sign/scale, filters must coincide.

    Caveat encoded below: filters within a near-degenerate eigen-subspace
    (evals ~ 0.5) are only defined up to rotation of that subspace, so they may
    NOT match individually even when both implementations are correct. The
    discriminative components — extreme evals near 0 and 1, i.e. the ones the
    pipeline actually uses — are well-separated and MUST match tightly. So this
    asserts on the extreme components and only warns on the middle.
    """
    mne_csp, mine, _, _ = fitted
    Wm, Wi = mne_csp.filters_, mine.filters_
    assert Wm.shape == Wi.shape, f"shape {Wi.shape} != MNE {Wm.shape}"

    ai, bj, cosv = _match_rows(Wm, Wi)

    # Rank MNE's filters by discriminative extremity: |eval - 0.5| large = good.
    extremity = np.abs(mne_csp.evals_[ai] - 0.5)
    order = np.argsort(-extremity)

    n_extreme = N_COMPONENTS          # the ones transform() keeps
    strong = order[:n_extreme]
    weak = order[n_extreme:]

    # Strong components: assert exact agreement, up to sign.
    for k in strong:
        i, j, c = ai[k], bj[k], cosv[k]
        assert abs(c) > 1 - 1e-4, \
            f"filter mismatch: MNE row {i} (eval={mne_csp.evals_[i]:.4f}) " \
            f"vs your row {j}, |cos|={abs(c):.6f}"
        sign = np.sign(c)
        wm = _unit_rows(Wm)[i]
        wi = _unit_rows(Wi)[j] * sign
        assert np.allclose(wm, wi, rtol=FILTER_RTOL, atol=1e-8), \
            f"strong filter {i}<->{j} aligned but not equal; " \
            f"max|Δ|={np.max(np.abs(wm - wi)):.2e}"

    # Weak/degenerate middle: don't fail, but surface how well they line up.
    if len(weak):
        worst = min(abs(cosv[k]) for k in weak)
        print(f"\n[info] degenerate-subspace filters, min|cos|={worst:.4f} "
              f"(low is expected near eval~0.5, not a bug)")


def test_accuracy_floor(data):
    """Floor, not proof. Swapping your CSP into the pipeline must reproduce the
    MNE end-to-end score within ~1 point. Catches gross plumbing errors; says
    nothing about whether the math is right (that's the filter test)."""
    X, y = data
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    def mean_acc(estimator):
        clf = Pipeline([("CSP", estimator), ("LDA", LinearDiscriminantAnalysis())])
        return cross_val_score(clf, X, y, cv=cv).mean()

    acc_mne = mean_acc(CSP(n_components=N_COMPONENTS, reg=None, log=True,
                           norm_trace=False))
    acc_mine = mean_acc(MyCSP(n_components=N_COMPONENTS, reg=None, log=True,
                              norm_trace=False))
    assert abs(acc_mne - acc_mine) <= ACC_FLOOR_PTS, \
        f"accuracy gap {abs(acc_mne - acc_mine):.4f} > {ACC_FLOOR_PTS} " \
        f"(mne={acc_mne:.4f}, mine={acc_mine:.4f})"
