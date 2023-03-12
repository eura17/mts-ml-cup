from __future__ import annotations

from typing import Any, Optional

import catboost as cb
import numpy as np
import pandas as pd
import polars as pl
from catboost.utils import get_gpu_device_count

from mts_ml_cup.validation import split_folds, calc_metrics
    

def fit(
    train: pl.DataFrame,
    pool_params: dict[str, Any],
    cb_params: dict[str, Any],
    verbose: int = 1_000,
    device: Optional[str] = None,
    sex_baseline: Optional[str] = None,
    age_baseline: Optional[list[str]] = None,
) -> tuple[list[cb.CatBoostClassifier], list[cb.CatBoostClassifier], list[dict[float]]]:
    if device is None:
        device = "GPU" if get_gpu_device_count() > 0 else "CPU"

    models_sex = []
    models_age = []
    metrics = []

    for i, (train_idx, val_idx) in enumerate(split_folds(train)):
        train_fold, val_fold = train[train_idx], train[val_idx]

        train_fold_sex = prepare_sex(train_fold)
        fold_train_sex = pool(
            train_fold_sex, 
            target_col="is_male", 
            baseline=train_fold_sex[sex_baseline] if sex_baseline is not None else None, 
            **pool_params,
        )
        val_fold_sex = prepare_sex(val_fold)
        fold_val_sex = pool(
            val_fold_sex, 
            target_col="is_male", 
            baseline=val_fold_sex[sex_baseline] if sex_baseline is not None else None,
            **pool_params,
        )
        fold_model_sex = cb.CatBoostClassifier(
            eval_metric="Logloss",
            loss_function="Logloss",
            allow_writing_files=False,
            task_type=device,
            **cb_params,
        )
        fold_model_sex.fit(fold_train_sex, eval_set=fold_val_sex, verbose=verbose)
        models_sex.append(fold_model_sex)
        
        train_fold_age = prepare_age(train_fold)
        fold_train_age = pool(
            train_fold_age, 
            target_col="age_bucket", 
            baseline=train_fold_age.select(age_baseline) if age_baseline is not None else None, 
            **pool_params,
        )
        val_fold_age = prepare_age(val_fold)
        fold_val_age = pool(
            val_fold_age, 
            target_col="age_bucket", 
            baseline=val_fold_age.select(age_baseline) if age_baseline is not None else None, 
            **pool_params,
        )
        fold_model_age = cb.CatBoostClassifier(
            eval_metric="MultiClass",
            loss_function="MultiClass",
            allow_writing_files=False,
            task_type=device,
            **cb_params,
        )
        fold_model_age.fit(fold_train_age, eval_set=fold_val_age, verbose=verbose)
        models_age.append(fold_model_age)

        fold_metrics = calc_metrics(
            is_male=fold_val_sex.get_label(),
            is_male_preds=fold_model_sex.predict_proba(fold_val_sex)[:, 1],
            age_bucket=fold_val_age.get_label(),
            age_bucket_preds=fold_model_age.predict(fold_val_age),
        )
        print()
        print(f"{'-' * 20} fold = {i} {'-' * 20}")
        for name, value in fold_metrics.items():
            print(f"{name} = {value:.4f}")
        print(f"{'-' * 20} fold = {i} {'-' * 20}")
        print()
        metrics.append(fold_metrics)
    
    return models_sex, models_age, metrics


def predict(
    test: pl.DataFrame,
    pool_params: dict[str, Any],
    models_sex: list[cb.CatBoostClassifier],
    models_age: list[cb.CatBoostClassifier],
) -> pd.DataFrame:
    test_pool = pool(test.select(pl.exclude("user_id")), **pool_params)

    is_male = 0
    for model_sex in models_sex:
        is_male += model_sex.predict_proba(test_pool)[:, 1] / len(models_sex)

    age_probas = 0
    for model_age in models_age:
        age_probas += model_age.predict_proba(test_pool) / len(models_age)
    age_bucket = np.argmax(age_probas, axis=1) + 1

    submission = pd.DataFrame()
    submission["user_id"] = test["user_id"].to_pandas()
    submission["is_male"] = is_male
    submission["age"] = age_bucket

    return submission


def pool(
    dataset: pl.DataFrame, 
    target_col: Optional[str] = None,
    baseline: Optional[pl.DataFrame] = None, 
    **pool_params,
) -> cb.Pool:
    return cb.Pool(
        data=(dataset.select(pl.exclude(target_col)) if target_col is not None else dataset).to_pandas(),
        label=dataset[target_col].to_pandas() if target_col is not None else None,
        baseline=baseline.to_numpy() if baseline is not None else None,
        **pool_params,
    )
    

def prepare_sex(train: pl.DataFrame) -> pl.DataFrame:
    return (
        train
        .filter(pl.col("is_male").is_not_null())
        .select(pl.exclude(["user_id", "age", "age_bucket"]))
    )


def prepare_age(train: pl.DataFrame) -> pl.DataFrame:
    return (
        train
        .filter(pl.col("age_bucket").is_not_null())
        .with_columns(pl.col("age_bucket").clip_min(1))
        .select(pl.exclude(["user_id", "is_male", "age"]))
    )
