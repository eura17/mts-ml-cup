import polars as pl
from mts_ml_cup.preprocessing import polars_map


def main(sessions: pl.DataFrame, url_cleaner) -> pl.DataFrame:
    sessions = (
        sessions
        .join(
            polars_map(
                mapping={
                    url: url_cleaner(url)
                    for url in sessions["url_host"].unique()
                },
                key_name="url_host",
                id_name="url_cleaned",
                id_dtype=pl.Utf8,
            ),
            on="url_host",
            how="left",
        )
        .select(pl.exclude("url_host"))
        .with_columns(pl.col("url_cleaned").alias("url_host"))
    )
    return (
        url_hosts_stats(sessions)
        .join(total_usage_stats(sessions), how="left", on="user_id")
        .join(usage_stats_per_date(sessions), how="left", on="user_id")
        .join(usage_stats_per_part_of_day(sessions), how="left", on="user_id")
        .join(usage_stats_per_session(sessions), how="left", on="user_id")
        .join(usage_stats_per_visit(sessions), how="left", on="user_id")
        .join(usage_stats_per_url(sessions), how="left", on="user_id")
        .join(usage_stats_per_url_daily(sessions), how="left", on="user_id")
        .join(usage_stats_per_url_partly(sessions), how="left", on="user_id")
        .join(part_of_day_distribution(sessions), how="left", on="user_id")
        .join(top_part_of_day(sessions), how="left", on="user_id")
    )


def url_hosts_stats(sessions: pl.DataFrame) -> pl.DataFrame:
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
                pl.col("chars_in_url").mean().alias("avg_chars_in_url"),
                pl.col("domains_in_url").mean().alias("avg_domains_in_url"),
                pl.col("is_yandex_turbo").mean().alias("yandex_turbo_share"),
                pl.col("is_google_turbo").mean().alias("google_turbo_share"),
                pl.col("is_mobile").mean().alias("mobile_share"),
            ]
            
        )
    )


def total_usage_stats(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id")
        .agg(
            [
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_days"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
                pl.struct(["date", "part_of_day_id"]).n_unique().alias("total_sessions"),
                pl.struct(["date", "part_of_day_id", "url_host"]).n_unique().alias("total_visits"),
            ]
        )
    )


def usage_stats_per_date(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date"])
        .agg(
            [
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_urls").mean().alias("avg_urls_per_day"),
                pl.col("total_requests").mean().alias("avg_requests_per_day"),
                pl.col("total_parts_of_day").mean().alias("avg_parts_of_day_per_day"),
            ]
        )
    )


def usage_stats_per_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(
            [
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_days"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_urls").mean().alias("avg_urls_per_part_of_day"),
                pl.col("total_requests").mean().alias("avg_requests_per_part_of_day"),
                pl.col("total_days").mean().alias("avg_days_per_part_of_day"),
            ]
        )
    )


def usage_stats_per_session(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date", "part_of_day_id"])
        .agg(
            [
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.col("request_cnt").sum().alias("total_requests"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_urls").mean().alias("avg_urls_per_session"),
                pl.col("total_requests").mean().alias("avg_requests_per_session"),
            ]
        )
    )


def usage_stats_per_visit(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date", "part_of_day_id", "url_host"])
        .agg(pl.col("request_cnt").sum().alias("total_requests"))
        .groupby("user_id")
        .agg(pl.col("total_requests").mean().alias("avg_requests_per_visit"))
    )


def usage_stats_per_url(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "url_host"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_days"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
                pl.struct(["date", "part_of_day_id"]).n_unique().alias("total_sessions"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("avg_requests_per_url"),
                pl.col("total_days").mean().alias("avg_days_per_url"),
                pl.col("total_parts_of_day").mean().alias("avg_parts_of_day_per_url"),
                pl.col("total_sessions").mean().alias("avg_sessions_per_url"),
            ]
        )
    )


def usage_stats_per_url_daily(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date", "url_host"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("avg_requests_per_url_daily"),
                pl.col("total_parts_of_day").mean().alias("avg_parts_of_day_per_url_daily"),
            ]
        )
    )


def usage_stats_per_url_partly(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id", "url_host"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_days"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("avg_requests_per_url_partly"),
                pl.col("total_days").mean().alias("avg_days_per_url_partly"),
            ]
        )
    )


def part_of_day_distribution(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(pl.col("request_cnt").sum().alias("requests"))
        .join(
            sessions
            .groupby("user_id")
            .agg(pl.col("request_cnt").sum().alias("total_requests")),
            on="user_id",
        )
        .select(
            [
                "user_id", 
                "part_of_day_id", 
                (pl.col("requests") / pl.col("total_requests")).alias("requests_share")
            ]
        )
        .pivot(
            values="requests_share",
            index="user_id",
            columns="part_of_day_id",
        )
        .rename(
            {
                str(part_of_day_id): f"part_of_day_{part_of_day_id}_requests_share"
                for part_of_day_id in range(1, 5)
            }
        )
        .fill_null(0)
    )


def top_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(pl.col("request_cnt").sum())
        .sort(["user_id", "request_cnt", "part_of_day_id"])
        .groupby("user_id")
        .agg(pl.col("part_of_day_id").last())
        .select(["user_id", pl.col("part_of_day_id").alias("top_part_of_day")])
    )
