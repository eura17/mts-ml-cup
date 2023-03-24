import itertools as it

import polars as pl


def urls_stats_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        .with_columns(
            [
                pl.col("url_host").str.lengths().alias("chars_in_url"),
                pl.col("url_host").apply(lambda url: len(url.split("."))).alias("domains_in_url"),
                pl.col("url_host").str.ends_with("turbopages.org").alias("is_yandex_turbo"),
                pl.col("url_host").str.ends_with("cdn.ampproject.org").alias("is_google_turbo"),
                pl.col("url_host").apply(lambda url: any(filter(lambda domain: domain == "m", url.split(".")))).alias("is_mobile"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("chars_in_url").mean().alias("url_avg_chars_in_url"),
                pl.col("domains_in_url").mean().alias("url_avg_domains_in_url"),
                pl.col("is_yandex_turbo").mean().alias("url_yandex_turbo_share"),
                pl.col("is_google_turbo").mean().alias("url_google_turbo_share"),
                pl.col("is_mobile").mean().alias("url_mobile_share"),
            ]
            
        )
    )


def top_n_urls_by_user(sessions: pl.DataFrame, top_n: int = 120) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "url_host"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "url_host"], descending=[False, True, False])
        .groupby("user_id")
        .head(top_n)
        .select(["user_id", "url_host"])
        .with_columns(pl.lit(1).alias("ones"))
        .select(
            [
                pl.all().exclude("ones"),
                pl.col("ones").cumsum().over("user_id").alias("top_url")
            ]
        )
        .select(pl.exclude("ones"))
        .pivot(
            index="user_id",
            columns="top_url",
            values="url_host",
        )
        .with_columns(pl.col(str(col)).fill_null("") for col in range(1, top_n + 1))
        .rename({str(col): f"url_top_{col}_url" for col in range(1, top_n + 1)})
    )


def all_urls_by_user_as_text(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "url_host"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "url_host"], descending=[False, True, False])
        .groupby("user_id")
        .agg(pl.col("url_host").apply(lambda urls: " ".join(urls)).alias("url_all_visited_urls"))
    )


def all_urls_combinations_as_text(sessions: pl.DataFrame, k: int = 2) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        .groupby("user_id")
        .agg(
            pl.col("url_host")
            .apply(lambda urls: " ".join(map("_+_".join, it.combinations(sorted(urls), k))))
            .alias(f"url_all_visited_urls_k_{k}")
        )
    )
