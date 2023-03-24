from __future__ import annotations

import polars as pl
from mts_ml_cup.utils import polars_map


def region_stats_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
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
                pl.col("region_id").n_unique().cast(pl.UInt8).alias("geo_regions_visited"),
                pl.col("region_id").last().cast(pl.UInt8).alias("geo_top_region_id"),
                pl.col("requests_share").last().cast(pl.Float32).alias("geo_top_region_share"),
            ]
        )
    )


def city_stats_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
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
                pl.col("city_id").n_unique().cast(pl.UInt16).alias("geo_cities_visited"),
                pl.col("city_id").last().cast(pl.UInt16).alias("geo_top_city_id"),
                pl.col("requests_share").last().cast(pl.Float32).alias("geo_top_city_share"),
            ]
        )
    )
