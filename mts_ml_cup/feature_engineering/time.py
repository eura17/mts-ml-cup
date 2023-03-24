import polars as pl


def time_period_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id")
        .agg(
            [
                pl.col("date").min().alias("first_day"),
                pl.col("date").max().alias("last_day"),
                pl.col("date").dt.truncate("1mo").min().alias("first_month"),
                pl.col("date").dt.truncate("1mo").max().alias("last_month"),
                pl.col("date").dt.truncate("1y").min().alias("first_year"),
                pl.col("date").dt.truncate("1y").max().alias("last_year"),
            ]
        )
        .select(
            [
                pl.col("user_id"),
                pl.col("first_day").cast(pl.Utf8).alias("time_first_day"),
                pl.col("last_day").cast(pl.Utf8).alias("time_last_day"),
                ((pl.col("last_day") - pl.col("first_day")).dt.days() + 1).alias("time_total_days"),
                pl.col("first_month").cast(pl.Utf8).alias("time_first_month"),
                pl.col("last_month").cast(pl.Utf8).alias("time_last_month"),
                (((pl.col("last_month") - pl.col("first_month")).dt.days() / 30).round(0) + 1).alias("time_total_months"),
                pl.col("first_year").cast(pl.Utf8).alias("time_first_year"),
                pl.col("last_year").cast(pl.Utf8).alias("time_last_year"),
                (((pl.col("last_year") - pl.col("first_year")).dt.days() / 365).round(0) + 1).alias("time_total_years"),
            ]
        )
    )


def top_part_of_day_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "part_of_day_id"])
        .groupby("user_id")
        .agg(pl.col("part_of_day_id").last())
        .select(["user_id", pl.col("part_of_day_id").alias("time_top_part_of_day")])
    )


def part_of_day_distribution_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(pl.col("request_cnt").sum())
        .join(
            sessions
            .groupby("user_id")
            .agg(pl.col("request_cnt").sum().alias("total_request_cnt")),
            on="user_id",
        )
        .select(
            [
                "user_id", 
                "part_of_day_id", 
                (pl.col("request_cnt") / pl.col("total_request_cnt")).alias("requests_share")
            ]
        )
        .pivot(
            values="requests_share",
            index="user_id",
            columns="part_of_day_id",
        )
        .rename(
            {
                str(part_of_day_id): f"time_part_of_day_{part_of_day_id}_requests_share"
                for part_of_day_id in sessions["part_of_day_id"].unique()
            }
        )
        .fill_null(0)
    )
