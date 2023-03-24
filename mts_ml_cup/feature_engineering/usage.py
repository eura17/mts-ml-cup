import polars as pl


def total_usage_stats_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id")
        .agg(
            [
                pl.col("request_cnt").sum().alias("usage_total_requests"),
                pl.col("date").n_unique().alias("usage_total_dates"),
                pl.col("part_of_day_id").n_unique().alias("usage_total_parts_of_day"),
                pl.col("url_host").n_unique().alias("usage_total_urls"),
                pl.struct(["date", "part_of_day_id"]).n_unique().alias("usage_total_sessions"),
                pl.struct(["date", "url_host"]).n_unique().alias("usage_total_daily_visits"),
                pl.struct(["part_of_day_id", "url_host"]).n_unique().alias("usage_total_partly_visits"),
                pl.struct(["date", "part_of_day_id", "url_host"]).n_unique().alias("usage_total_visits"),
            ]
        )
    )


def usage_stats_per_date(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.struct(["part_of_day_id", "url_host"]).n_unique().alias("total_partly_visits"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("usage_avg_requests_per_date"),
                pl.col("total_parts_of_day").mean().alias("usage_avg_parts_of_day_per_date"),
                pl.col("total_urls").mean().alias("usage_avg_urls_per_date"),
                pl.col("total_partly_visits").mean().alias("usage_avg_partly_visits"),
            ]
        )
    )


def usage_stats_per_part_of_day(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_dates"),
                pl.col("url_host").n_unique().alias("total_urls"),
                pl.struct(["date", "url_host"]).n_unique().alias("total_daily_visits"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("usage_avg_requests_per_part_of_day"),
                pl.col("total_dates").mean().alias("usage_avg_dates_per_part_of_day"),
                pl.col("total_urls").mean().alias("usage_avg_urls_per_part_of_day"),
                pl.col("total_daily_visits").mean().alias("usage_avg_daily_visits_per_part_of_day"),
            ]
        )
    )


def usage_stats_per_url(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "url_host"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_dates"),
                pl.col("part_of_day_id").n_unique().alias("total_parts_of_day"),
                pl.struct(["date", "part_of_day_id"]).n_unique().alias("total_sessions"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("avg_requests_per_url"),
                pl.col("total_dates").mean().alias("avg_dates_per_url"),
                pl.col("total_parts_of_day").mean().alias("avg_parts_of_day_per_url"),
                pl.col("total_sessions").mean().alias("avg_sessions_per_url"),
            ]
        )
    )


def usage_stats_per_session(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "date", "part_of_day_id"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("url_host").n_unique().alias("total_urls"),
                
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("usage_avg_requests_per_session"),
                pl.col("total_urls").mean().alias("usage_avg_urls_per_session"),
            ]
        )
    )


def usage_stats_per_daily_visit(sessions: pl.DataFrame) -> pl.DataFrame:
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
                pl.col("total_requests").mean().alias("usage_avg_requests_per_daily_visit"),
                pl.col("total_parts_of_day").mean().alias("usage_avg_parts_of_day_per_daily_visit"),
            ]
        )
    )


def usage_stats_per_partly_visit(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby(["user_id", "part_of_day_id", "url_host"])
        .agg(
            [
                pl.col("request_cnt").sum().alias("total_requests"),
                pl.col("date").n_unique().alias("total_dates"),
            ]
        )
        .groupby("user_id")
        .agg(
            [
                pl.col("total_requests").mean().alias("usage_avg_requests_per_partly_visit"),
                pl.col("total_dates").mean().alias("usage_avg_dates_per_partly_visit"),
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
