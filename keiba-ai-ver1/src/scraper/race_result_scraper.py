"""
レース結果のスクレイピング
netkeiba.com の個別レースページから結果テーブルを取得し Parquet で保存する
"""
import logging
import re
from pathlib import Path

import pandas as pd

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class RaceResultScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = self.cfg["base_url"]
        self.out_dir = Path(config["paths"]["raw_races"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def scrape(self, race_id: str) -> pd.DataFrame | None:
        """1レースのデータを取得して DataFrame で返す（同期版）"""
        url = f"{self.base_url}/race/{race_id}/"
        soup = self.get(url)
        if soup is None:
            return None
        return self.parse(soup, race_id)

    def parse(self, soup, race_id: str) -> pd.DataFrame | None:
        """fetch 済みの soup から DataFrame を生成する（非同期版から呼ばれる）"""
        race_meta = self._parse_race_meta(soup, race_id)
        if race_meta is None:
            return None

        result_table = soup.find("table", class_="race_table_01")
        if result_table is None:
            logger.warning(f"Result table not found: {race_id}")
            return None

        rows = []
        for tr in result_table.find_all("tr")[1:]:
            row = self._parse_result_row(tr)
            if row:
                row.update(race_meta)
                rows.append(row)

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["race_id"] = race_id
        return df

    def _parse_race_meta(self, soup, race_id: str) -> dict | None:
        try:
            # race_name: class なしの h1（空でない最初のもの）
            race_name_text = next(
                (h1.get_text(strip=True) for h1 in soup.find_all("h1") if h1.get_text(strip=True)),
                ""
            )

            # data_intro に全情報が入っている
            # 例: "1 R2歳未勝利芝右1800m / 天候 : 曇 / 芝 : 良  / 発走 : 09:55 2020年07月25日"
            data_intro = soup.find("div", class_="data_intro")
            track_info = ""
            if data_intro:
                track_info = data_intro.get_text(separator=" ", strip=True)

            course_type = "芝" if "芝" in track_info else "ダート"
            distance = self._extract_int(r"(\d{3,4})m", track_info)
            weather = self._extract_str(r"天候\s*:\s*(\S+)", track_info)
            # "芝 : 良" または "ダート : 良" のフォーマット
            track_condition = self._extract_str(r"(?:芝|ダート)\s*:\s*(\S+)", track_info)
            direction = "右" if "右" in track_info else ("左" if "左" in track_info else "")

            # race_date: data_intro 末尾の "2020年07月25日" を取得
            race_date = ""
            date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", track_info)
            if date_match:
                y, m, d = date_match.groups()
                race_date = f"{y}-{int(m):02d}-{int(d):02d}"

            # フォールバック: ul.race_place から日付を取得
            if not race_date:
                race_place = soup.find("ul", class_="race_place")
                if race_place:
                    date_match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", race_place.get_text())
                    if date_match:
                        y, m, d = date_match.groups()
                        race_date = f"{y}-{int(m):02d}-{int(d):02d}"

            return {
                "race_name": race_name_text,
                "course_type": course_type,
                "distance": distance,
                "weather": weather,
                "track_condition": track_condition,
                "direction": direction,
                "race_date": race_date,
                "venue_code": race_id[4:6],
            }
        except Exception as e:
            logger.error(f"Failed to parse race meta for {race_id}: {e}")
            return None

    def _parse_result_row(self, tr) -> dict | None:
        tds = tr.find_all("td")
        if len(tds) < 18:
            return None
        try:
            # 着順（数字以外は除外: 取消・除外等）
            finish_str = tds[0].get_text(strip=True)
            if not finish_str.isdigit():
                return None

            horse_a = tds[3].find("a")
            horse_id = ""
            if horse_a and horse_a.get("href"):
                m = re.search(r"/horse/(\d+)/", horse_a["href"])
                if m:
                    horse_id = m.group(1)

            jockey_a = tds[6].find("a")
            jockey_id = ""
            if jockey_a and jockey_a.get("href"):
                m = re.search(r"/jockey/result/recent/(\w+)/", jockey_a["href"])
                if m:
                    jockey_id = m.group(1)

            trainer_a = tds[18].find("a") if len(tds) > 18 else None
            trainer_id = ""
            if trainer_a and trainer_a.get("href"):
                m = re.search(r"/trainer/result/recent/(\w+)/", trainer_a["href"])
                if m:
                    trainer_id = m.group(1)

            weight_str = tds[14].get_text(strip=True)  # 例: "480(+4)"
            horse_weight, weight_diff = self._parse_weight(weight_str)

            return {
                "finish": int(finish_str),
                "gate": self._safe_int(tds[1].get_text(strip=True)),
                "horse_num": self._safe_int(tds[2].get_text(strip=True)),
                "horse_name": tds[3].get_text(strip=True),
                "horse_id": horse_id,
                "sex_age": tds[4].get_text(strip=True),       # 例: "牡3"
                "weight_carried": self._safe_float(tds[5].get_text(strip=True)),
                "jockey_name": tds[6].get_text(strip=True),
                "jockey_id": jockey_id,
                "time": tds[7].get_text(strip=True),
                "margin": tds[8].get_text(strip=True),
                "popularity": self._safe_int(tds[10].get_text(strip=True)),
                "odds": self._safe_float(tds[9].get_text(strip=True)),
                "last3f": self._safe_float(tds[11].get_text(strip=True)),
                "corner_positions": tds[12].get_text(strip=True),
                "trainer_name": tds[18].get_text(strip=True) if len(tds) > 18 else "",
                "trainer_id": trainer_id,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
            }
        except Exception as e:
            logger.debug(f"Row parse error: {e}")
            return None

    def _parse_weight(self, s: str):
        m = re.match(r"(\d+)\(([+-]?\d+)\)", s)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    def _extract_int(self, pattern: str, text: str) -> int | None:
        m = re.search(pattern, text)
        return int(m.group(1)) if m else None

    def _extract_str(self, pattern: str, text: str) -> str:
        m = re.search(pattern, text)
        return m.group(1) if m else ""

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

    def save(self, df: pd.DataFrame, race_id: str):
        path = self.out_dir / f"{race_id}.parquet"
        df.to_parquet(path, index=False)

    def load_all(self) -> pd.DataFrame:
        files = list(self.out_dir.glob("*.parquet"))
        if not files:
            return pd.DataFrame()
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
