"""
出馬表のスクレイピング（未来レース向け）
race.netkeiba.com の shutuba ページから出走馬情報を取得する。
結果ページが存在しない開催前レースの predict に使用する。
"""
import logging
import re

import pandas as pd

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class ShutubaScraper(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.race_url = self.cfg.get("race_url", "https://race.netkeiba.com")

    def scrape(self, race_id: str) -> pd.DataFrame | None:
        url = f"{self.race_url}/race/shutuba.html?race_id={race_id}"
        soup = self.get(url, fresh_session=True)
        if soup is None:
            return None
        return self.parse(soup, race_id)

    def parse(self, soup, race_id: str) -> pd.DataFrame | None:
        race_meta = self._parse_race_meta(soup, race_id)
        if race_meta is None:
            return None

        table = soup.find("table", class_="Shutuba_Table")
        if table is None:
            logger.warning(f"Shutuba_Table not found: {race_id}")
            return None

        rows = []
        for i, tr in enumerate(table.find_all("tr"), start=0):
            tds = tr.find_all("td")
            if len(tds) < 8:
                continue
            row = self._parse_row(tds, i)
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
            # タイトルから日付を抽出: "... | 2026年4月12日 ..."
            title = soup.title.get_text(strip=True) if soup.title else ""
            race_date = ""
            m = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", title)
            if m:
                y, mo, d = m.groups()
                race_date = f"{y}-{int(mo):02d}-{int(d):02d}"

            data01 = soup.find(class_="RaceData01")
            track_info = data01.get_text(separator=" ", strip=True) if data01 else ""

            course_type = "芝" if "芝" in track_info else "ダート"
            distance_m = re.search(r"(\d{3,4})m", track_info)
            distance = int(distance_m.group(1)) if distance_m else None
            direction = "右" if "右" in track_info else ("左" if "左" in track_info else "")

            return {
                "race_name": soup.find(class_="RaceName").get_text(strip=True) if soup.find(class_="RaceName") else "",
                "course_type": course_type,
                "distance": distance,
                "weather": "",           # 開催前は不明
                "track_condition": "",   # 開催前は不明
                "direction": direction,
                "race_date": race_date,
                "venue_code": race_id[4:6],
            }
        except Exception as e:
            logger.error(f"Failed to parse shutuba meta for {race_id}: {e}")
            return None

    def _parse_row(self, tds, row_index: int) -> dict | None:
        try:
            # Waku/Umaban は開催前は空のことがある → row_index で代用
            gate_text = tds[0].get_text(strip=True)
            gate = int(gate_text) if gate_text.isdigit() else None
            horse_num_text = tds[1].get_text(strip=True)
            horse_num = int(horse_num_text) if horse_num_text.isdigit() else row_index

            horse_a = tds[3].find("a")
            if not horse_a:
                return None
            horse_name = horse_a.get_text(strip=True)
            horse_id = ""
            m = re.search(r"/horse/(\d+)", horse_a.get("href", ""))
            if m:
                horse_id = m.group(1)

            sex_age = tds[4].get_text(strip=True)
            weight_carried = self._safe_float(tds[5].get_text(strip=True))

            jockey_a = tds[6].find("a")
            jockey_name = tds[6].get_text(strip=True)
            jockey_id = ""
            if jockey_a:
                m = re.search(r"/jockey/result/recent/(\w+)/", jockey_a.get("href", ""))
                if m:
                    jockey_id = m.group(1)

            trainer_a = tds[7].find("a")
            trainer_name = tds[7].get_text(strip=True)
            trainer_id = ""
            if trainer_a:
                m = re.search(r"/trainer/result/recent/(\w+)/", trainer_a.get("href", ""))
                if m:
                    trainer_id = m.group(1)

            weight_text = tds[8].get_text(strip=True)  # 例: "480(+2)" or ""
            horse_weight, weight_diff = self._parse_weight(weight_text)

            return {
                "finish": None,
                "gate": gate,
                "horse_num": horse_num,
                "horse_name": horse_name,
                "horse_id": horse_id,
                "sex_age": sex_age,
                "weight_carried": weight_carried,
                "jockey_name": jockey_name,
                "jockey_id": jockey_id,
                "time": None,
                "margin": None,
                "popularity": None,
                "odds": None,
                "last3f": None,
                "corner_positions": None,
                "trainer_name": trainer_name,
                "trainer_id": trainer_id,
                "horse_weight": horse_weight,
                "weight_diff": weight_diff,
            }
        except Exception as e:
            logger.debug(f"Shutuba row parse error: {e}")
            return None

    def _parse_weight(self, s: str):
        m = re.match(r"(\d+)\(([+-]?\d+)\)", s)
        if m:
            return int(m.group(1)), int(m.group(2))
        return None, None

    def _safe_float(self, s: str) -> float | None:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None
