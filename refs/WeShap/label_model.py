import numpy as np
import optuna.logging
from wrench.labelmodel import MeTaL, MajorityVoting, Snorkel

optuna.logging.disable_default_handler()


def get_wrench_label_model(method, **kwargs):
    if method == "Snorkel":
        return Snorkel(**kwargs)
    elif method == "MeTaL":
        return MeTaL(**kwargs)
    elif method == "MV":
        return MajorityVoting(**kwargs)


def is_valid_snorkel_input(lfs):
    lf_labels = [lf.label for lf in lfs]
    return len(lfs) > 3 and np.min(lf_labels) != np.max(lf_labels)


def is_valid_snorkel_L(L):
    unique_labels = np.unique(L.reshape(-1))
    return len(unique_labels)>=3 and L.shape[1] > 3
