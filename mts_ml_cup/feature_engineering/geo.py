import polars as pl


def unique_regions(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id").agg(pl.col("region_name").n_unique().alias("n_regions"))
    )


def the_most_popular_region(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "region_name"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "region_name"])
        .groupby("user_id")
        .agg(pl.col("region_name").last())
        .select(["user_id", pl.col("region_name").alias("top_region")])
    )


def unique_cities(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "region_name", "city_name"])
        .unique()
        ["user_id"].value_counts()
        .select(["user_id", pl.col("counts").alias("n_cities")])
    )


def the_most_popular_city(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "region_name", "city_name"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "city_name", "region_name"])
        .groupby("user_id")
        .agg([pl.col("region_name").last(), pl.col("city_name").last()])
        .select(["user_id", pl.concat_str([pl.col("region_name"), pl.col("city_name")], sep=" + ").alias("top_city")])
    )
