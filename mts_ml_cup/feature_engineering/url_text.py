import polars as pl


def main(sessions: pl.DataFrame, url_cleaner) -> pl.DataFrame:
    return (
        urls_text(sessions, url_cleaner)
        .join(urls_tokens(sessions, url_cleaner), how="left", on="user_id")
    )



def urls_text(sessions: pl.DataFrame, url_cleaner) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        .with_columns(pl.col("url_host").apply(url_cleaner))
        .groupby("user_id")
        .agg(pl.col("url_host").unique().alias("urls"))
        .with_columns(pl.col("urls").apply(lambda urls: " ".join(urls)).alias("urls_text"))
        .select(["user_id", "urls_text"])
    )


def urls_tokens(sessions: pl.DataFrame, url_cleaner) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "url_host"])
        .unique()
        .with_columns(pl.col("url_host").apply(url_cleaner).str.split(".").alias("url_part"))
        .select(["user_id", "url_part"])
        .explode("url_part")
        .groupby("user_id")
        .agg(pl.col("url_part").unique().alias("url_parts"))
        .with_columns(pl.col("url_parts").apply(lambda urls: " ".join(urls)).alias("urls_tokens"))
        .select(["user_id", "urls_tokens"])
    )
