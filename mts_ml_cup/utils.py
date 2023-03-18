from __future__ import annotations

import bisect

import pandas as pd
import polars as pl


def age_to_bucket(age: int) -> int:
    return bisect.bisect_left([18, 25, 35, 45, 55, 65], age)


def polars_map(
    mapping: dict[str, int], 
    key_name: str, 
    id_name: str,
    id_dtype: pl.DataType,
) -> pl.DataFrame:
    return (
        pl.from_pandas(
            pd.Series(mapping)
            .reset_index(drop=False)
            .rename(columns={"index": key_name, 0: id_name})
        )
        .with_columns(pl.col(id_name).cast(id_dtype))
    )
