import polars as pl


def unique_urls(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        ["user_id"].value_counts()
        .select(["user_id", pl.col("counts").alias("n_urls")])
    )


def avg_url_length(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        .groupby("user_id")
        .agg(pl.col("url_host").str.lengths().mean().alias("avg_url_length"))
    )
