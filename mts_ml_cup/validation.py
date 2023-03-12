from __future__ import annotations

from typing import Generator

import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import KFold


def split_folds(
    x: pd.DataFrame, 
    n_splits: int = 5,
) -> Generator:
    return KFold(n_splits=n_splits, shuffle=True, random_state=777).split(x)


def calc_metrics(
    is_male: pd.Series,
    is_male_preds: pd.Series,
    age_bucket: pd.Series,
    age_bucket_preds: pd.DataFrame,
) -> dict[str, float]:
    sex_roc_auc = roc_auc_score(is_male, is_male_preds)
    age_f1 = f1_score(age_bucket, age_bucket_preds, average="weighted")
    return {
        "sex ROC-AUC": sex_roc_auc,
        "age F1 Weighted": age_f1,
        "mts-ml-cup metric": (2 * sex_roc_auc - 1) + 2 * age_f1,
    }
