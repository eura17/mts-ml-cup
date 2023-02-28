import polars as pl


def days_share_by_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day"])
        .agg(pl.col("date").n_unique().alias("n_unique_days_by_part_of_day"))
        .join(
            sessions
            .groupby("user_id")
            .agg(pl.col("date").n_unique().alias("n_unique_days")),
            on="user_id",
        )
        .select(
            [
                "user_id", 
                "part_of_day", 
                (pl.col("n_unique_days_by_part_of_day") / pl.col("n_unique_days")).alias("days_share")
            ]
        )
        .pivot(
            values="days_share",
            index="user_id",
            columns="part_of_day"
        )
        .rename(
            {
                part_of_day: f"{part_of_day}_days_share"
                for part_of_day in ["evening", "night", "morning", "day"]
            }
        )
        .fill_null(0)
    )


def urls_share_by_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day"])
        .agg(pl.col("url_host").n_unique().alias("n_unique_urls_by_part_of_day"))
        .join(
            sessions
            .groupby("user_id")
            .agg(pl.col("url_host").n_unique().alias("n_unique_urls")),
            on="user_id",
        )
        .select(
            [
                "user_id", 
                "part_of_day", 
                (pl.col("n_unique_urls_by_part_of_day") / pl.col("n_unique_urls")).alias("urls_share")
            ]
        )
        .pivot(
            values="urls_share",
            index="user_id",
            columns="part_of_day"
        )
        .rename(
            {
                part_of_day: f"{part_of_day}_urls_share"
                for part_of_day in ["evening", "night", "morning", "day"]
            }
        )
        .fill_null(0)
    )


def requests_share_by_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day"])
        .agg(pl.col("request_cnt").sum().alias("requests_by_part_of_day"))
        .join(
            sessions
            .groupby("user_id")
            .agg(pl.col("request_cnt").sum().alias("requests")),
            on="user_id",
        )
        .select(
            [
                "user_id", 
                "part_of_day", 
                (pl.col("requests_by_part_of_day") / pl.col("requests")).alias("requests_share")
            ]
        )
        .pivot(
            values="requests_share",
            index="user_id",
            columns="part_of_day"
        )
        .rename(
            {
                part_of_day: f"{part_of_day}_requests_share"
                for part_of_day in ["evening", "night", "morning", "day"]
            }
        )
        .fill_null(0)
    )
