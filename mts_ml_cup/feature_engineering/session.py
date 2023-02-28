from __future__ import annotations

import polars as pl


def total_requests(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id")
        .agg(pl.col("request_cnt").sum().alias("total_requests"))
    )


def n_sessions(sessions: pl.DataFrame, group: list[str], name: str) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id"] + group)
        .unique()
        ["user_id"].value_counts()
        .select(["user_id", pl.col("counts").alias(name)])
    )


def avg_unique_urls_by_session(sessions: pl.DataFrame, group: list[str], name: str) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id"] + group)
        .agg(pl.col("url_host").n_unique().alias("n_urls"))
        .groupby("user_id")
        .agg(pl.col("n_urls").mean().alias(name))
    )


def avg_requests_by_session(sessions: pl.DataFrame, group: list[str], name: str) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id"] + group)
        .agg(pl.col("request_cnt").sum())
        .groupby("user_id")
        .agg(pl.col("request_cnt").mean().alias(name))
    )


def avg_day_parts_by_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "date", "part_of_day"])
        .unique()
        .groupby(["user_id", "date"])
        .agg(pl.col("part_of_day").n_unique().alias("n_parts"))
        .groupby("user_id")
        .agg(pl.col("n_parts").mean().alias("avg_day_parts_by_day"))
    )


def the_most_popular_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "part_of_day"])
        .groupby("user_id")
        .agg(pl.col("part_of_day").last())
        .select(["user_id", pl.col("part_of_day").alias("top_part_of_day")])
    )
