from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ShiftResult:
    name: str
    roc_auc: float
    pr_auc: float
    sensitivity: float
    specificity: float


def add_gaussian_noise(X: pd.DataFrame, sigma: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    noise = rng.normal(loc=0.0, scale=sigma, size=Xn.shape)
    Xn.iloc[:, :] = Xn.to_numpy() + noise
    return Xn


def drift_feature(X: pd.DataFrame, feature: str, shift_in_std: float) -> pd.DataFrame:
    """
    Add a systematic shift to one feature: x := x + shift_in_std * std(feature)
    """
    Xd = X.copy()
    std = float(Xd[feature].std())
    Xd[feature] = Xd[feature] + shift_in_std * std
    return Xd


def subpopulation_slice(X: pd.DataFrame, y: np.ndarray, feature: str, quantile: float = 0.8):
    """
    Return the slice where feature >= quantile(feature) (harder tail slice).
    """
    thr = float(X[feature].quantile(quantile))
    mask = X[feature] >= thr
    return X.loc[mask], y[mask], thr
