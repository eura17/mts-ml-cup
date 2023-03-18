from __future__ import annotations

import pandas as pd
import pypdfium2 as pdfium
from tqdm import tqdm


def parse_rosstat(
    path: str,
    start_page: int = 63,
    pages_per_region: int = 4,
) -> pd.DataFrame:
    rosstat = pdfium.PdfDocument(path)
    regions_stats = []
    for first_page_idx in tqdm(range(start_page, len(rosstat), pages_per_region)):
        rosstat_region_pages = [rosstat[first_page_idx + offset] for offset in range(pages_per_region)]
        region_stats = parse_region_stats(rosstat_region_pages)
        regions_stats.append(region_stats)
    return pd.concat(regions_stats).reset_index(drop=True)


def parse_region_stats(pages: list[pdfium.PdfPage]) -> pd.DataFrame:
    page_text_rows = [p.get_textpage().get_text_range().split("\n") for p in pages]
    stats = pd.concat(
        [
            parse_first_page(page_text_rows[0]),
            parse_middle_page(page_text_rows[1]),
            parse_middle_page(page_text_rows[2]),
            parse_last_page(page_text_rows[3]),
        ]
    ).reset_index(drop=True)
    stats["region"] = parse_region_name(page_text_rows[0])
    return stats.loc[
        :, 
        [
            "region", "age", 
            "men", "women", 
            "urban_men", "urban_women", 
            "rural_men", "rural_women",
        ]
    ]


def parse_first_page(rows: list[str]) -> pd.DataFrame:
    return parse_table(rows[10:-6])


def parse_middle_page(rows: list[str]) -> pd.DataFrame:
    return parse_table(rows[11:])


def parse_last_page(rows: list[str]) -> pd.DataFrame:
    stats = parse_table(rows[11:-6])
    
    eighty_plus_values = rows[-6].split()
    men = as_population(eighty_plus_values[4])
    women = as_population(eighty_plus_values[5])
    urban_men = as_population(eighty_plus_values[7])
    urban_women = as_population(eighty_plus_values[8])
    rural_men = as_population(eighty_plus_values[10])
    rural_women = as_population(eighty_plus_values[11])
    eighty_plus_stats = pd.DataFrame(
        [[80, men, women, urban_men, urban_women, rural_men, rural_women]],
        columns=stats.columns,
    )
    
    return pd.concat([stats, eighty_plus_stats]).reset_index(drop=True)


def parse_table(rows: list[str]) -> pd.DataFrame:
    stats = []
    for row in rows:
        row_values = row.strip().split()
        
        if row_values[1] == "–":
            continue
        
        age = int(row_values[0])
        men = as_population(row_values[2])
        women = as_population(row_values[3])
        urban_men = as_population(row_values[5])
        urban_women = as_population(row_values[6])
        rural_men = as_population(row_values[8])
        rural_women = as_population(row_values[9])
        
        stats.append([age, men, women, urban_men, urban_women, rural_men, rural_women])

    return pd.DataFrame(stats, columns=["age", "men", "women", "urban_men", "urban_women", "rural_men", "rural_women"])


def as_population(pop: str) -> int:
    if pop == "-":
        return 0
    return int(pop)


def parse_region_name(first_page_rows: list[str]) -> str:
    return first_page_rows[-4].strip()


if __name__ == "__main__":
    from mts_ml_cup.utils import age_to_bucket

    stats = parse_rosstat("~/mts-ml-cup/data/raw/rosstat.pdf")
    stats.insert(loc=2, column="age_bucket", value=stats["age"].apply(age_to_bucket))
    stats.to_csv("~/mts-ml-cup/data/processed/rosstat.csv", index=False)
