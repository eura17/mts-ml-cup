from __future__ import annotations

import polars as pl
from mts_ml_cup.preprocessing import polars_map

PRICES = {
    55: 2899.0,  # Atlas LLC_+_G450
    59: 12500.0,  # Blackview_+_BV6000
    60: 19900.0,  # Blackview_+_BV9500
    61: 10299.0,  # Doke Communication (HK) Limited_+_BV5900
    62: 9900.0,  # Doke Communication (HK) Limited_+_BV6100
    63: 16990.0,  # Doke Communication (HK) Limited_+_BV6800 Pro
    64: 17450.0,  # Doke Communication (HK) Limited_+_BV9100
    65: 18900.0,  # Doke Communication (HK) Limited_+_BV9500 Plus
    66: 21990.0,  # Doke Communication (HK) Limited_+_BV9600 Pro
    67: 19990.0,  # Doogee_+_S60
    68: 21490.0,  # Doogee_+_S68Pro
    69: 49000.0,  # Google Inc_+_Pixel 2 XL
    70: 53500.0,  # Google Inc_+_Pixel 3
    71: 52000.0,  # Google Inc_+_Pixel 4
    72: 32500.0,  # Google Inc_+_Pixel 4a
    73: 48000.0,  # Google Inc_+_Pixel 5
    178: 2495.0,  # Itel Technology Limited_+_A16 Plus
    179: 4290.0,  # Itel Technology Limited_+_L5002
    180: 5999.0,  # Itel Technology Limited_+_L6006
    461: 6999.0,  # Tecno_+_Camon 15 Air
    462: 5995.0,  # Tecno_+_LC7
    463: 6990.0,  # Tecno_+_SPARK 4
    464: 6499.0,  # Tecno_+_SPARK 6 Go
    465: 8990.0,  # Tecno_+_Spark 5 Air
    466: 15870.0,  # Umi Network Technology Co Limited_+_BISON
    467: 8995.0,  # Vingroup Joint Stock Company_+_Joy 4
    468: 6995.0,  # Vingroup Joint Stock Company_+_V430
    588: 13990.0,  # Yandex LLC_+_YNDX-000SB
}


def main(sessions: pl.DataFrame, prices: dict[int, float] = PRICES) -> pl.DataFrame:
    return (
        manufacturer_by_user(sessions)
        .with_columns(
            pl.when(pl.col("manufacturer_id") == 2)  # apple
            .then(1)
            .otherwise(2)
            .cast(pl.UInt8)
            .alias("os_id")
        )
        .join(model_by_user(sessions), how="outer", on="user_id")
        .join(price_by_model(sessions, prices), how="outer", on="model_id")
    )


def manufacturer_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "manufacturer_id"])
        .unique()
    )


def model_by_user(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .select(["user_id", "model_id"])
        .unique()
    )


def price_by_model(sessions: pl.DataFrame, prices: dict[int, float]) -> pl.DataFrame:
    model_prices = (
        sessions
        .select(["model_id", "manufacturer_id", "price"])
        .unique()
        .groupby(["manufacturer_id", "model_id"])
        .agg(pl.col("price").mean().cast(pl.Float32).alias("avg_price"))
    )
    manufacturer_prices = (
        model_prices
        .groupby("manufacturer_id")
        .agg(pl.col("avg_price").mean().cast(pl.Float32).alias("avg_price"))
    )
    model_prices = (
        model_prices
        .join(manufacturer_prices, how="left", on="manufacturer_id", suffix="_manufacturer")
        .select(["model_id", pl.coalesce(["avg_price", "avg_price_manufacturer"]).alias("avg_price")])
    )
    return (
        model_prices
        .join(
            other=polars_map(prices, key_name="model_id", id_name="price", id_dtype=pl.Float32)
                .with_columns(pl.col("model_id").cast(pl.UInt16)),
            how="left",
            on="model_id",
        )
        .select(["model_id", pl.coalesce(["avg_price", "price"]).alias("avg_price")])
    )
