from __future__ import annotations

import pandas as pd
import polars as pl

MANUFACTURER_MAP = {
    "Motorola": ["Motorola", "Motorola Mobility LLC, a Lenovo Company"],
    "Google": ["Google Inc"],
    "Itel": ["Itel Technology Limited"],
    "Oppo": ["Oppo"],
    "Realme": ["Realme Chongqing Mobile Telecommunications Corp Ltd", "Realme Mobile Telecommunications (Shenzhen) Co Ltd"],
    "OnePlus": ["OnePlus"],
    "Honor": ["Honor Device Company Limited"],
    "Asus": ["Asus"],
    "Atlas": ["Atlas LLC"],
    "ZTE": ["ZTE"],
    "Highscreen": ["Highscreen Limited", "Highscreen"],
    "Samsung": ["Samsung"],
    "Huawei": ["Huawei", "Huawei Device Company Limited"],
    "HTC": ["HTC"],
    "Blackview": ["Blackview"],
    "Vivo": ["Vivo"],
    "Sony": ["Sony", "Sony Mobile Communications Inc."],
    "LeEco": ["LeEco"],
    "Lenovo": ["Lenovo"],
    "Apple": ["Apple"],
    "Nokia": ["Nokia"],
    "LG": ["LG"],
    "Tecno": ["Tecno"],
    "Doogee": ["Doogee"],
    "BQ": ["BQ Devices Limited"],
    "Doke": ["Doke Communication (HK) Limited"],
    "Yandex": ["Yandex LLC"],
    "Umi": ["Umi Network Technology Co Limited"],
    "Vsmart": ["Vingroup Joint Stock Company"],
    "Meizu": ["Meizu"],
    "Xiaomi": ["Xiaomi"],
    "Alcatel": ["Alcatel"],
}


def device_manufacturer(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .join(
            _build_joinable_map(
                mapping=MANUFACTURER_MAP,
                map_from="cpe_manufacturer_name",
                map_to="device_manufacturer",
            ), 
            how="left", 
            on="cpe_manufacturer_name",
        )
        .select(["user_id", "device_manufacturer"])
        .unique()
    )


def device_model(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .join(
            _build_joinable_map(
                mapping=MANUFACTURER_MAP,
                map_from="cpe_manufacturer_name",
                map_to="device_manufacturer",
            ), 
            how="left", 
            on="cpe_manufacturer_name",
        )
        .select(["user_id", "device_manufacturer", "cpe_model_name"])
        .unique()
        .select(["user_id", pl.concat_str(["device_manufacturer", "cpe_model_name"], sep=" + ").alias("device_model")])
    )


def device_mean_price(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions
        .groupby("user_id")
        .agg(pl.col("price").mean())
        .select(["user_id", pl.col("price").alias("device_mean_price")])
    )


def _build_joinable_map(mapping: dict[str, list[str]], map_from: str, map_to: str) -> pl.DataFrame:
    return pl.from_pandas(
        pd.Series(mapping)
        .reset_index()
        .rename(columns={"index": map_to, 0: map_from})
        .explode(map_from)
)
