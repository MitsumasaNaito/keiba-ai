"""
馬の過去成績スクレイピング
netkeiba.com の馬個別ページから全出走歴を取得する
"""
import logging
import re
from pathlib import Path

import pandas as pd

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class HorseHistoryScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = self.cfg["base_url"]
        self.out_dir = Path(config["paths"]["raw_horses"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def scrape(self, horse_id: str) -> pd.DataFrame | None:
        """馬の過去成績を取得して DataFrame で返す（同期版）"""
        url = f"{self.base_url}/horse/result/{horse_id}/"
        soup = self.get(url)
        if soup is None:
            return None
        return self.parse(soup, horse_id)

    def parse(self, soup, horse_id: str) -> pd.DataFrame | None:
        """fetch 済みの soup から DataFrame を生成する（非同期版から呼ばれる）"""
        table = soup.find("table", class_="db_h_race_results")
        if table is None:
            logger.warning(f"History table not found for horse: {horse_id}")
            return None

        rows = []
        for tr in table.find_all("tr")[1:]:
            row = self._parse_row(tr)
            if row:
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["horse_id"] = horse_id
        return df

    def _parse_row(self, tr) -> dict | None:
        tds = tr.find_all("td")
        if len(tds) < 22:
            return None
        try:
            finish_str = tds[11].get_text(strip=True)
            if not finish_str.isdigit():
                return None

            race_a = tds[4].find("a")
            race_id = ""
            if race_a and race_a.get("href"):
                m = re.search(r"/race/(\d{12})/", race_a["href"])
                if m:
                    race_id = m.group(1)

            return {
                "race_date": tds[0].get_text(strip=True),
                "venue": tds[1].get_text(strip=True),
                "weather": tds[2].get_text(strip=True),
                "race_num": self._safe_int(tds[3].get_text(strip=True)),
                "race_name": tds[4].get_text(strip=True),
                "race_id": race_id,
                "field_size": self._safe_int(tds[6].get_text(strip=True)),
                "gate": self._safe_int(tds[7].get_text(strip=True)),
                "horse_num": self._safe_int(tds[8].get_text(strip=True)),
                "odds": self._safe_float(tds[9].get_text(strip=True)),
                "popularity": self._safe_int(tds[10].get_text(strip=True)),
                "finish": int(finish_str),
                "jockey": tds[12].get_text(strip=True),
                "weight_carried": self._safe_float(tds[13].get_text(strip=True)),
                "distance_type": tds[14].get_text(strip=True),  # 例: "芝1600"
                "track_condition": tds[15].get_text(strip=True),
                "time": tds[17].get_text(strip=True),
                "margin": tds[18].get_text(strip=True),
                "last3f": self._safe_float(tds[22].get_text(strip=True)) if len(tds) > 22 else None,
                "horse_weight": self._safe_int(tds[23].get_text(strip=True)) if len(tds) > 23 else None,
            }
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            return None

    def _safe_int(self, s: str) -> int | None:
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    def _safe_float(self, s: str) -> float | None:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def save(self, df: pd.DataFrame, horse_id: str):
        path = self.out_dir / f"{horse_id}.parquet"
        df.to_parquet(path, index=False)

    def load_horse(self, horse_id: str) -> pd.DataFrame | None:
        path = self.out_dir / f"{horse_id}.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)
