import numpy as np
import pickle

import resources as R


def load_data():
    idx_pho   = np.load(R.IDX_PHONEMES)
    idx_w     = np.load(R.IDX_WORDS)
    with open(R.DATA_CTL, 'rb') as f:
        datactl = pickle.load(f)

    return datactl, idx_pho, idx_w
