"""
Parquetファイルからの生データ読み込み
"""
from pathlib import Path

import pandas as pd


class RawLoader:
    def __init__(self, config: dict):
        self.raw_races_dir = Path(config["paths"]["raw_races"])
        self.raw_horses_dir = Path(config["paths"]["raw_horses"])

    def load_races(self) -> pd.DataFrame:
        """全レース結果を結合して返す"""
        files = sorted(self.raw_races_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df = self._parse_race_dates(df)
        df = self._parse_sex_age(df)
        return df

    def load_horse(self, horse_id: str) -> pd.DataFrame | None:
        path = self.raw_horses_dir / f"{horse_id}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def load_all_horses(self) -> pd.DataFrame:
        files = sorted(self.raw_horses_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    def _parse_race_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        if "race_date" in df.columns:
            df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
        return df

    def _parse_sex_age(self, df: pd.DataFrame) -> pd.DataFrame:
        """sex_age 列（例: "牡3"）を sex と age に分割"""
        if "sex_age" not in df.columns:
            return df
        df["sex"] = df["sex_age"].str[0]
        df["age"] = pd.to_numeric(df["sex_age"].str[1:], errors="coerce")
        return df
