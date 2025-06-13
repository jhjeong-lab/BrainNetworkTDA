import os 
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import datasets, image
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker
from nilearn.glm.first_level import FirstLevelModel

# === Fetch AAL atlas ===
aal = datasets.fetch_atlas_aal()
aal_img = aal.maps        # path to AAL atlas NIfTI file
aal_labels = aal.labels   # list of ROI names

# === Parcellation ===
masker = NiftiLabelsMasker(labels_img   = aal_img,
                           standardize  = 'zscore_sample',
                           detrend      = True,
                           low_pass     = None,
                           high_pass    = None,
                           t_r          = 0.72,
                           resampling_target = 'data')   # keep atlas grid

BASE_DIR = Path("../Dataset")

def make_fmri_path(subject_id: int,
                   run_label: str = "REST1_LR",
                   hp_tag: str = "hp2000_clean") -> Path:
    
    return (BASE_DIR /
            f"{subject_id:06d}" /
            "MNINonLinear" /
            "Results" /
            f"rfMRI_{run_label}" /
            f"rfMRI_{run_label}_{hp_tag}.nii.gz")

def load_MRI_data(subject_id: int, 
                  run_label: str,
                  trim: int,
                  win_size: int,
                  step_size: int):
    
    fMRI_Path = make_fmri_path(subject_id, run_label)
    
    if not fMRI_Path.exists():
        print(f"오류: 파일을 찾을 수 없습니다 -> {fMRI_Path}")
        return None

    ts = masker.fit_transform(str(fMRI_Path))
    ts_trimmed = ts[trim:]
    print("Extracted time-series shape:", ts_trimmed.shape)
    
    connectivity = ConnectivityMeasure(kind="correlation")
    T, P = ts_trimmed.shape
    n_windows = (T - win_size) // step_size + 1
    out = np.zeros((n_windows, P, P), dtype=np.float32)
    for idx, s in enumerate(range(0, T - win_size + 1, step_size)):
        segment = ts_trimmed[s:s+win_size]
        corr_matrix = connectivity.fit_transform([segment])[0]
        out[idx] = corr_matrix
    
    return out

def vectorize_upper_triangular(matrices):
    """
    Convert a batch of symmetric (P×P) matrices into vectors of upper triangle (excluding diag).
    Input:  matrices.shape = (N, P, P)
    Output: vectors.shape  = (N, P*(P-1)/2)
    """
    N, P, _ = matrices.shape
    idx = np.triu_indices(P, k=1)
    vecs = matrices[:, idx[0], idx[1]]
    return vecs

