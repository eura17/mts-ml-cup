from __future__ import annotations

import json
from typing import Any, Callable, Optional
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd
import polars as pl
from catboost.utils import get_gpu_device_count

from mts_ml_cup.modeling.validation import kfold_split, calc_metrics


class CatBoostCV:
    def __init__(
        self,
        model_params: dict[str, Any],
        pool_params: dict[str, Any],
        splitter: Callable[[pl.DataFrame], tuple[np.ndarray, np.ndarray]] = kfold_split,
    ) -> None:
        model_params.setdefault("task_type", "GPU" if get_gpu_device_count() > 0 else "CPU")
        model_params["allow_writing_files"] = False

        self.model_params = model_params
        self.pool_params = pool_params
        self.splitter = splitter

    def fit(self, train: pl.DataFrame, verbose: int = 1_000) -> list[dict[str, float]]:
        self.models_sex_ = []
        self.models_age_ = []
        self.metrics_ = []

        for fold, (train_idx, val_idx) in enumerate(self.splitter(train)):
            train_fold, val_fold = train[train_idx], train[val_idx]

            train_fold_sex_pool = self.to_pool_sex(train_fold)
            val_fold_sex_pool = self.to_pool_sex(val_fold)
            fold_model_sex = cb.CatBoostClassifier(eval_metric="Logloss", loss_function="Logloss", **self.model_params)
            fold_model_sex.fit(train_fold_sex_pool, eval_set=val_fold_sex_pool, verbose=verbose)
            self.models_sex_.append(fold_model_sex)
            
            fold_train_age = self.to_pool_age(train_fold)
            fold_val_age = self.to_pool_age(val_fold)
            fold_model_age = cb.CatBoostClassifier(eval_metric="MultiClass", loss_function="MultiClass", **self.model_params)
            fold_model_age.fit(fold_train_age, eval_set=fold_val_age, verbose=verbose)
            self.models_age_.append(fold_model_age)

            fold_metrics = calc_metrics(
                is_male=val_fold_sex_pool.get_label(),
                is_male_preds=fold_model_sex.predict_proba(val_fold_sex_pool)[:, 1],
                age_bucket=fold_val_age.get_label(),
                age_bucket_preds=fold_model_age.predict(fold_val_age),
            )
            print()
            print(f"{'-' * 20} {fold = } {'-' * 20}")
            for name, value in fold_metrics.items():
                print(f"{name} = {value:.4f}")
            print(f"{'-' * 20} {fold = } {'-' * 20}")
            print()
            self.metrics_.append(fold_metrics)
        
        return self.metrics_

    def predict(self, test: pl.DataFrame, fold: Optional[int] = None) -> pd.DataFrame:
        if fold is not None:
            models_sex = [self.models_sex_[fold]]
            models_age = [self.models_age_[fold]]
        else:
            models_sex = self.models_sex_
            models_age = self.models_age_

        test_pool = self.to_pool(test)
        is_male = np.mean([model.predict_proba(test_pool)[:, 1] for model in models_sex], axis=0)
        age_probas = np.mean([model.predict_proba(test_pool) for model in models_age], axis=0)
    
        pred = pd.DataFrame()
        pred.loc[:, "user_id"] = test["user_id"].to_pandas()
        pred.loc[:, "is_male"] = is_male
        pred.loc[:, [f"age_bucket_{i}_proba" for i in range(1, age_probas.shape[1] + 1)]] = age_probas
        pred.loc[:, "age"] = np.argmax(age_probas, axis=1) + 1

        return pred

    def predict_oof(self, train: pl.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                self.predict(train[val_idx], fold=fold).assign(fold=fold)
                for fold, (_, val_idx) in enumerate(self.splitter(train))
            ]
        ).reset_index(drop=True)

    def feature_importances(self, task: str) -> pd.Series:
        models = getattr(self, f"models_{task}_")
        fi = 0
        for model in models:
            fi += pd.Series(model.feature_importances_, index=model.feature_names_) / len(models)
        return fi.sort_values(ascending=False)

    def to_pool(self, dataset: pl.DataFrame) -> cb.Pool:
        return cb.Pool(
            data=dataset.select(pl.exclude(["user_id", "age", "age_bucket", "is_male"])).to_pandas(),
            **self.pool_params,
        )

    def to_pool_sex(self, dataset: pl.DataFrame) -> cb.Pool:
        dataset = (
            dataset
            .filter(pl.col("is_male").is_not_null())
            .select(pl.exclude(["user_id", "age", "age_bucket"]))
        )
        return cb.Pool(
            data=dataset.select(pl.exclude("is_male")).to_pandas(),
            label=dataset["is_male"].to_pandas(),
            **self.pool_params,
        )

    def to_pool_age(self, dataset: pl.DataFrame) -> cb.Pool:
        dataset = (
            dataset
            .filter(pl.col("age_bucket").is_not_null())
            .with_columns(pl.col("age_bucket").clip_min(1))
            .select(pl.exclude(["user_id", "is_male", "age"]))
        )
        return cb.Pool(
            data=dataset.select(pl.exclude("age_bucket")).to_pandas(),
            label=dataset["age_bucket"].to_pandas(),
            **self.pool_params,
        )

    def save_models(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for fold, (model_sex, model_age) in enumerate(zip(self.models_sex_, self.models_age_)):
            fold_path = path / f"fold-{fold}"
            fold_path.mkdir(parents=True, exist_ok=True)
            model_sex.save_model(str(fold_path / "sex.cbm"))
            model_age.save_model(str(fold_path / "age.cbm"))
        with open(path / "metrics.json", "w") as f:
            json.dump(self.metrics_, f)

    @classmethod
    def from_snapshot(cls, path: str | Path, **kwargs) -> CatBoostCV:
        model = cls(**kwargs)

        path = Path(path)
        folds = [p for p in path.iterdir() if p.name.startswith("fold")]

        models_sex = []
        models_age = []
        for fold in sorted(folds):
            fold_model_sex = cb.CatBoostClassifier()
            fold_model_sex.load_model(str(fold / "sex.cbm"))
            models_sex.append(fold_model_sex)

            fold_model_age = cb.CatBoostClassifier()
            fold_model_age.load_model(str(fold / "age.cbm"))
            models_age.append(fold_model_age)
        
        with open(path / "metrics.json", "r") as f:
            metrics = json.load(f)

        model.models_sex_ = models_sex
        model.models_age_ = models_age
        model.metrics_ = metrics
        return model
