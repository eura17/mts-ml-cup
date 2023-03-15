from __future__ import annotations

from typing import Any, Optional, Callable

import catboost as cb
import numpy as np
import pandas as pd
import polars as pl
from catboost.utils import get_gpu_device_count

from mts_ml_cup.validation import kfold_split, calc_metrics
    

def fit(
    train: pl.DataFrame,
    pool_params: dict[str, Any],
    model_params: dict[str, Any],
    splitter: Callable[[pl.DataFrame], tuple[np.ndarray, np.ndarray]] = kfold_split,
    verbose: int = 1_000,
) -> tuple[list[cb.CatBoostClassifier], list[cb.CatBoostClassifier], list[dict[float]]]:
    model_params.setdefault("task_type", "GPU" if get_gpu_device_count() > 0 else "CPU")
    model_params["allow_writing_files"] = False

    models_sex = []
    models_age = []
    metrics = []

    for i, (train_idx, val_idx) in enumerate(splitter(train)):
        train_fold, val_fold = train[train_idx], train[val_idx]

        train_fold_sex_pool = pool_from_polars(prepare_sex(train_fold), "is_male", **pool_params)
        val_fold_sex_pool = pool_from_polars(prepare_sex(val_fold), "is_male", **pool_params)
        fold_model_sex = cb.CatBoostClassifier(eval_metric="Logloss", loss_function="Logloss", **model_params)
        fold_model_sex.fit(train_fold_sex_pool, eval_set=val_fold_sex_pool, verbose=verbose)
        models_sex.append(fold_model_sex)
        
        fold_train_age = pool_from_polars(prepare_age(train_fold), "age_bucket", **pool_params)
        fold_val_age = pool_from_polars(prepare_age(val_fold), "age_bucket", **pool_params)
        fold_model_age = cb.CatBoostClassifier(eval_metric="MultiClass", loss_function="MultiClass", **model_params)
        fold_model_age.fit(fold_train_age, eval_set=fold_val_age, verbose=verbose)
        models_age.append(fold_model_age)

        fold_metrics = calc_metrics(
            is_male=val_fold_sex_pool.get_label(),
            is_male_preds=fold_model_sex.predict_proba(val_fold_sex_pool)[:, 1],
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
    test_pool = pool_from_polars(test.select(pl.exclude("user_id")), **pool_params)

    is_male = 0
    for model_sex in models_sex:
        is_male += model_sex.predict_proba(test_pool)[:, 1] / len(models_sex)

    age_probas = 0
    for model_age in models_age:
        age_probas += model_age.predict_proba(test_pool) / len(models_age)
    age_bucket = np.argmax(age_probas, axis=1) + 1

    pred = pd.DataFrame()
    pred.loc[:, "user_id"] = test["user_id"].to_pandas()
    pred.loc[:, "is_male"] = is_male
    pred.loc[:, [f"age_bucket_proba_{i}" for i in range(1, age_probas.shape[1] + 1)]] = age_probas
    pred.loc[:, "age"] = age_bucket

    return pred


def pool_from_polars(dataset: pl.DataFrame, target_col: Optional[str] = None, **kwargs) -> cb.Pool:
    return cb.Pool(
        data=(dataset.select(pl.exclude(target_col)) if target_col is not None else dataset).to_pandas(),
        label=dataset[target_col].to_pandas() if target_col is not None else None,
        **kwargs,
    )


def prepare_sex(dataset: pl.DataFrame) -> pl.DataFrame:
    return (
        dataset
        .filter(pl.col("is_male").is_not_null())
        .select(pl.exclude(["user_id", "age", "age_bucket"]))
    )


def prepare_age(dataset: pl.DataFrame) -> pl.DataFrame:
    return (
        dataset
        .filter(pl.col("age_bucket").is_not_null())
        .with_columns(pl.col("age_bucket").clip_min(1))
        .select(pl.exclude(["user_id", "is_male", "age"]))
    )
