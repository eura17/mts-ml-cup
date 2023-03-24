from __future__ import annotations

import polars as pl


def region_stats(stats: pl.DataFrame) -> pl.DataFrame:
    return (
        stats
        .groupby("region_id")
        .agg(
            [
                ((pl.col("men") + pl.col("women")).sum()).alias("total_people"),
                ((pl.col("urban_men") + pl.col("urban_women")).sum()).alias("total_urban_people"),
            ]
        )
        .filter(pl.col("region_id").is_not_null())
        .select(
            [
                "region_id",
                pl.col("total_people").alias("rosstat_total_people"),
                (pl.col("total_urban_people") / pl.col("total_people")).alias("rosstat_urban_people_share"),
            ]
        )
    )


def sex_share_by_region(stats: pl.DataFrame) -> pl.DataFrame:
    return (
        stats
        .groupby("region_id")
        .agg(
            (pl.col("men").sum() / (pl.col("men") + pl.col("women")).sum()).alias("rosstat_men_share"),
        )
        .filter(pl.col("region_id").is_not_null())
    )


def age_share_by_region(stats: pl.DataFrame) -> pl.DataFrame:
    return (
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
        .with_columns(
            pl.sum([pl.col(str(i)) for i in range(1, 7)]).alias("total_people")
        )
        .with_columns(
            [
                (pl.col(str(i)) / pl.col("total_people")).alias(f"{i}_share")
                for i in range(1, 7)
            ]
        )
        .select(["region_id"] + [pl.col(f"{i}_share").alias(f"rosstat_age_bucket_{i}_share") for i in range(1, 7)])
    )


def sex_age_share_by_region(stats: pl.DataFrame) -> pl.DataFrame:
    return (
        stats
        .with_columns(pl.col("age_bucket").clip_min(1))
        .groupby(["region_id", "age_bucket"])
        .agg(
            [
                (pl.col("men").sum() / (pl.col("men") + pl.col("women")).sum()).alias("rosstat_men_age_bucket"),
                (pl.col("women").sum() / (pl.col("men") + pl.col("women")).sum()).alias("rosstat_women_age_bucket"),
            ]
        )
        .filter(pl.col("region_id").is_not_null())
        .pivot(
            values=["rosstat_men_age_bucket", "rosstat_women_age_bucket"],
            index="region_id",
            columns="age_bucket",
        )
        .select(
            ["region_id"]
            + [pl.col(f"rosstat_men_age_bucket_{i}").alias(f"rosstat_men_age_bucket_{i}_share") for i in range(1, 7)]
            + [pl.col(f"rosstat_women_age_bucket_{i}").alias(f"rosstat_women_age_bucket_{i}_share") for i in range(1, 7)]
        )
    )
