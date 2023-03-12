from __future__ import annotations

import bisect

import pandas as pd
import polars as pl


def prepare_sessions(
    part_path: str,
    regions_mapping: dict[str, int],
    cities_mapping: dict[str, int],
    manufacturers_mapping: dict[str, int],
    models_mapping: dict[str, int],
    parts_of_day_mapping: dict[str, int],
) -> pl.DataFrame:
    return (
        pl.read_parquet(part_path)
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
                "url_host",
                "price",
                "date",
                "part_of_day_id",
                "request_cnt",
                "user_id",
            ]
        )
    )


def prepare_train(path: str) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(
            pl.col("user_id").cast(pl.UInt32),
            pl.col("is_male").apply(lambda s: s if s != "NA" else pl.Null).cast(pl.UInt8),
            pl.col("age").cast(pl.UInt16),
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


def prepare_test(path: str) -> pl.DataFrame:
    return pl.read_parquet(path).with_columns(pl.col("user_id").cast(pl.UInt32)).select(["user_id"])


def age_to_bucket(age: int) -> int:
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], age)


def polars_map(
    mapping: dict[str, int], 
    key_name: str, 
    id_name: str,
    id_dtype: pl.DataType,
) -> pl.DataFrame:
    return (
        pl.from_pandas(
            pd.Series(mapping)
            .reset_index(drop=False)
            .rename(columns={"index": key_name, 0: id_name})
        )
        .with_columns(pl.col(id_name).cast(id_dtype))
    )
