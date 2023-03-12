from __future__ import annotations

import polars as pl
from mts_ml_cup.preprocessing import polars_map


def main(
    sessions: pl.DataFrame,
    rosstat_stats: pl.DataFrame,
    regions_mapping: dict[str, int],
) -> pl.DataFrame:
    return (
        regions(sessions)
        .join(cities(sessions), how="outer", on="user_id")
        .join(rosstat(rosstat_stats, regions_mapping), how="left", left_on="main_region_id", right_on="region_id")
    )


def regions(sessions: pl.DataFrame) -> pl.DataFrame:
    user_region_requests = (
        sessions
        .groupby(["user_id", "region_id"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "region_id"])
    )
    user_region_requests = (
        user_region_requests
        .join(
            other=user_region_requests
                .groupby("user_id")
                .agg(pl.col("request_cnt").sum().alias("total_requests")),
            how="left",
            on="user_id",
        )
        .with_columns((pl.col("request_cnt") / pl.col("total_requests")).alias("requests_share"))
    )
    return (
        user_region_requests
        .groupby("user_id")
        .agg(
            [
                pl.col("region_id").n_unique().cast(pl.UInt8).alias("regions_visited"),
                pl.col("region_id").last().cast(pl.UInt8).alias("main_region_id"),
                pl.col("requests_share").last().cast(pl.Float32).alias("main_region_share"),
            ]
        )
    )


def cities(sessions: pl.DataFrame) -> pl.DataFrame:
    user_city_requests = (
        sessions
        .groupby(["user_id", "city_id"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "city_id"])
    )
    user_city_requests = (
        user_city_requests
        .join(
            other=user_city_requests
                .groupby("user_id")
                .agg(pl.col("request_cnt").sum().alias("total_requests")),
            how="left",
            on="user_id",
        )
        .with_columns((pl.col("request_cnt") / pl.col("total_requests")).alias("requests_share"))
    )
    return (
        user_city_requests
        .groupby("user_id")
        .agg(
            [
                pl.col("city_id").n_unique().cast(pl.UInt16).alias("cities_visited"),
                pl.col("city_id").last().cast(pl.UInt16).alias("main_city_id"),
                pl.col("requests_share").last().cast(pl.Float32).alias("main_city_share"),
            ]
        )
    )


def rosstat(stats: pl.DataFrame, regions_mapping: dict[str, int]) -> pl.DataFrame:
    stats = stats.join(
        other=polars_map(regions_mapping, key_name="region_mts", id_name="region_id", id_dtype=pl.UInt8),
        how="left",
        on="region_mts",
    )

    sex = (
        stats
        .groupby("region_id")
        .agg(
            [
                (pl.col("men").sum() / (pl.col("men") + pl.col("women")).sum()).alias("men_share"),
                (pl.col("women").sum() / (pl.col("men") + pl.col("women")).sum()).alias("women_share"),
            ]
        )
        .filter(pl.col("region_id").is_not_null())
    )

    age = (
        stats
        .with_columns(pl.col("age_bucket").clip_min(1))
        .groupby(["region_id", "age_bucket"])
        .agg(
            [
                (pl.col("men") + pl.col("women")).sum().alias("people")
            ]
        )
        .filter(pl.col("region_id").is_not_null())
        .pivot(
            values="people",
            index="region_id",
            columns="age_bucket",
        )
        .rename({str(i): f"age_bucket_{i}" for i in range(1, 7)})
        .with_columns(
            (
                pl.col("age_bucket_1") 
                + pl.col("age_bucket_2") 
                + pl.col("age_bucket_3")
                + pl.col("age_bucket_4")
                + pl.col("age_bucket_5")
                + pl.col("age_bucket_6")
            ).alias("total")
        )
        .with_columns(
            [
                (pl.col(f"age_bucket_{i}") / pl.col("total")).alias(f"age_bucket_{i}_share")
                for i in range(1, 7)
            ]
        )
        .select(["region_id"] + [f"age_bucket_{i}_share" for i in range(1, 7)])
    )

    sex_age = (
        stats
        .with_columns(pl.col("age_bucket").clip_min(1))
        .groupby(["region_id", "age_bucket"])
        .agg(
            [
                (pl.col("men").sum() / (pl.col("men") + pl.col("women")).sum()).alias("men_share"),
                (pl.col("women").sum() / (pl.col("men") + pl.col("women")).sum()).alias("women_share"),
            ]
        )
        .filter(pl.col("region_id").is_not_null())
        .pivot(
            values=["men_share", "women_share"],
            index="region_id",
            columns="age_bucket",
        )
    )

    return (
        sex
        .join(age, how="outer", on="region_id")
        .join(sex_age, how="outer", on="region_id")
        .filter(pl.col("region_id").is_not_null())
    )
