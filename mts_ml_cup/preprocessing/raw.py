from __future__ import annotations

import os

import polars as pl
from tqdm import tqdm

from mts_ml_cup.utils import age_to_bucket, polars_map


def find_unique_cat_variables(parts_path: str) -> list[set]:
    regions = set()
    cities = set()
    manufacturers = set()
    models = set()
    types = set()
    oss = set()
    parts_of_day = set()

    parts_path = "../data/raw/competition_data_final_pqt/"
    for p in tqdm(os.listdir(parts_path)):
        if not p.endswith(".parquet"):
            continue
        
        part = (
            pl.read_parquet(os.path.join(parts_path, p))
            .with_columns(
                [
                    pl.concat_str([pl.col("region_name"), pl.col("city_name")], sep="_+_").alias("city_name"),
                    pl.concat_str([pl.col("cpe_manufacturer_name"), pl.col("cpe_model_name")], sep="_+_").alias("cpe_model_name"),
                ]
            )
        )
        regions |= set(part["region_name"].unique())
        cities |= set(part["city_name"].unique())
        manufacturers |= set(part["cpe_manufacturer_name"].unique())
        models |= set(part["cpe_model_name"].unique())
        types |= set(part["cpe_type_cd"].unique())
        oss |= set(part["cpe_model_os_type"].unique())
        parts_of_day |= set(part["part_of_day"].unique())

    return regions, cities, manufacturers, models, types, oss, parts_of_day


def convert_sessions(
    sessions: pl.DataFrame,
    regions_mapping: dict[str, int],
    cities_mapping: dict[str, int],
    manufacturers_mapping: dict[str, int],
    models_mapping: dict[str, int],
    types_mapping: dict[str, int],
    os_mapping: dict[str, int],
    parts_of_day_mapping: dict[str, int],
) -> pl.DataFrame:
    return (
        sessions
        .with_columns(
            [
                pl.concat_str([pl.col("region_name"), pl.col("city_name")], sep="_+_").alias("city_name"),
                pl.concat_str([pl.col("cpe_manufacturer_name"), pl.col("cpe_model_name")], sep="_+_").alias("cpe_model_name"),
                pl.col("price").cast(pl.Float32),
                pl.col("request_cnt").cast(pl.UInt8),
                pl.col("user_id").cast(pl.UInt32),
            ]
        )
        .join(
            other=polars_map(regions_mapping, "region_name", "region_id", pl.UInt8),
            how="left",
            on="region_name",
        )
        .join(
            other=polars_map(cities_mapping, "city_name", "city_id", pl.UInt16),
            how="left",
            on="city_name",
        )
        .join(
            other=polars_map(manufacturers_mapping, "cpe_manufacturer_name", "manufacturer_id", pl.UInt8),
            how="left",
            on="cpe_manufacturer_name",
        )
        .join(
            other=polars_map(models_mapping, "cpe_model_name", "model_id", pl.UInt16),
            how="left",
            on="cpe_model_name",
        )
        .join(
            other=polars_map(types_mapping, "cpe_type_cd", "type_id", pl.UInt8),
            how="left",
            on="cpe_type_cd",
        )
        .join(
            other=polars_map(os_mapping, "cpe_model_os_type", "os_id", pl.UInt8),
            how="left",
            on="cpe_model_os_type",
        )
        .join(
            other=polars_map(parts_of_day_mapping, "part_of_day", "part_of_day_id", pl.UInt8),
            how="left",
            on="part_of_day",
        )
        .select(
            [
                "region_id",
                "city_id",
                "manufacturer_id",
                "model_id",
                "type_id",
                "os_id",
                "url_host",
                "price",
                "date",
                "part_of_day_id",
                "request_cnt",
                "user_id",
            ]
        )
    )


def convert_train(train: pl.DataFrame) -> pl.DataFrame:
    return (
        train
        .with_columns(
            pl.col("user_id").cast(pl.UInt32),
            pl.col("is_male").apply(lambda s: s if s != "NA" else pl.Null).cast(pl.UInt8),
            pl.col("age").cast(pl.UInt8),
            pl.col("age").apply(age_to_bucket).cast(pl.UInt8).alias("age_bucket"),
        )
        .select(
            [
                "user_id",
                "is_male",
                "age",
                "age_bucket",
            ]
        )
    )


def convert_test(test: pl.DataFrame) -> pl.DataFrame:
    return test.with_columns(pl.col("user_id").cast(pl.UInt32)).select(["user_id"])
