"""
レースID一覧の収集
netkeiba.com の月別 → 日別 の2段階でrace_idを取得する

URL構造:
  /race/list/YYYYMM/   → 開催日リスト (/race/list/YYYYMMDD/)
  /race/list/YYYYMMDD/ → レースID一覧 (/race/XXXXXXXXXXXX/)

キャッシュ:
  data/race_id_cache/YYYYMM.txt に月ごとのレースIDを保存する
  ファイルが存在する月はHTTPリクエストをスキップする
"""
import logging
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

RACE_ID_PATTERN = re.compile(r"(?:/race/(?:[a-z]+/)?|race_id=)(\d{12})")
DATE_PAGE_PATTERN = re.compile(r"/race/list/(\d{8})/")


class RaceIdCollector(BaseScraper):
    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = self.cfg["base_url"]
        self.cache_dir = Path(config["paths"]["race_id_cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # venue_codes でフィルタ（地方競馬など不正IDを除外）
        self._venue_codes = set(self.cfg.get("venue_codes", [
            "01","02","03","04","05","06","07","08","09","10"
        ]))

    def _is_valid_race_id(self, race_id: str) -> bool:
        """JRA venue code (01〜10) のIDのみ有効とする"""
        return len(race_id) == 12 and race_id[4:6] in self._venue_codes

    def _cache_path(self, year: int, month: int) -> Path:
        return self.cache_dir / f"{year:04d}{month:02d}.txt"

    def _load_cache(self, year: int, month: int) -> list[str] | None:
        """キャッシュが存在すれば読み込む。なければ None を返す"""
        path = self._cache_path(year, month)
        if not path.exists():
            return None
        ids = [line for line in path.read_text().splitlines() if line]
        logger.debug(f"{year}/{month:02d}: キャッシュから {len(ids)} レース読み込み")
        return ids

    def _save_cache(self, year: int, month: int, race_ids: list[str]):
        """レースIDをキャッシュファイルに保存する"""
        path = self._cache_path(year, month)
        path.write_text("\n".join(race_ids))

    def collect_month(self, year: int, month: int) -> list[str]:
        """
        指定年月のレースID一覧を取得する
        キャッシュがあればHTTPリクエストをスキップする
        """
        # キャッシュヒット
        cached = self._load_cache(year, month)
        if cached is not None:
            return [rid for rid in cached if self._is_valid_race_id(rid)]

        # キャッシュなし → HTTPで取得
        month_url = f"{self.base_url}/race/list/{year:04d}{month:02d}/"
        soup = self.get(month_url, fresh_session=True)
        if soup is None:
            return []

        # 開催日ページのリンクを収集
        date_pages = []
        for a_tag in soup.find_all("a", href=DATE_PAGE_PATTERN):
            m = DATE_PAGE_PATTERN.search(a_tag["href"])
            if m:
                date_pages.append(m.group(1))
        date_pages = list(set(date_pages))

        if not date_pages:
            self._save_cache(year, month, [])
            return []

        # 各開催日ページからレースIDを並列取得
        max_workers = self.cfg.get("concurrent_requests", 5)
        race_ids = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._fetch_day_ids, date_str): date_str
                for date_str in sorted(date_pages)
            }
            for future in as_completed(futures):
                race_ids.extend(future.result())

        race_ids = sorted(set(rid for rid in race_ids if self._is_valid_race_id(rid)))
        self._save_cache(year, month, race_ids)
        # tqdm.write を使うことでプログレスバーと干渉しない
        tqdm.write(
            f"  {year}/{month:02d}: {len(date_pages)}開催日, {len(race_ids)}レース"
        )
        return race_ids

    def collect_years(self, years: list[int], until: str | None = None) -> list[str]:
        """
        複数年度のレースID一覧を取得する
        キャッシュ済みの月はHTTPリクエストをスキップする
        """
        until_year, until_month = self._parse_until(until)

        targets = [
            (y, m)
            for y in years
            for m in range(1, 13)
            if not (y > until_year or (y == until_year and m > until_month))
        ]

        # キャッシュ済みと未収集に分類
        cached_targets  = [(y, m) for y, m in targets if self._cache_path(y, m).exists()]
        pending_targets = [(y, m) for y, m in targets if not self._cache_path(y, m).exists()]
        logger.info(
            f"対象: {len(targets)}ヶ月 "
            f"(キャッシュ済み: {len(cached_targets)}, 未収集: {len(pending_targets)})"
        )

        all_ids = []

        # キャッシュ済みはまとめて即時読み込み（進捗バーなし）
        for year, month in cached_targets:
            all_ids.extend(self.collect_month(year, month))

        # 未収集のみ進捗バーを表示して取得
        if pending_targets:
            with tqdm(pending_targets, desc="レースID収集", unit="月", dynamic_ncols=True,
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
                for year, month in pbar:
                    pbar.set_postfix({"取得中": f"{year}/{month:02d}"})
                    ids = self.collect_month(year, month)
                    all_ids.extend(ids)

        return sorted(set(all_ids))

    def _fetch_day_ids(self, date_str: str) -> list[str]:
        """1開催日分のレースIDを取得する（スレッドごとに独立したセッションを使用）"""
        url = f"{self.base_url}/race/list/{date_str}/"
        # スレッドセーフのため独立したセッションを作成
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.cfg["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })
        time.sleep(random.uniform(
            self.cfg["request_interval_min"],
            self.cfg["request_interval_max"],
        ))
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url}")
                return []
            resp.encoding = resp.apparent_encoding
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            return []

        ids = []
        for a_tag in soup.find_all("a", href=RACE_ID_PATTERN):
            m = RACE_ID_PATTERN.search(a_tag["href"])
            if m:
                ids.append(m.group(1))
        return ids

    def _parse_until(self, until: str | None) -> tuple[int, int]:
        if until is None:
            return (9999, 12)
        parts = until.split("-")
        return (int(parts[0]), int(parts[1]))

    def fetch_day_ids_listing(self, date_str: str) -> list[str]:
        """
        指定日のレースID一覧を取得する（list_races コマンド用）。
        - 過去日付: db.netkeiba.com の日別ページから取得（高速・安定）
        - 当日以降: race.netkeiba.com の2段階APIから取得（出馬表リンクを解析）
        """
        from datetime import date
        today_str = date.today().strftime("%Y%m%d")

        if date_str <= today_str:
            # db.netkeiba.com の日別ページ（BaseScraper.get() のリトライ付き）
            url = f"{self.base_url}/race/list/{date_str}/"
            soup = self.get(url, fresh_session=True)
            if soup is None:
                return []
            ids = []
            for a_tag in soup.find_all("a", href=RACE_ID_PATTERN):
                m = RACE_ID_PATTERN.search(a_tag["href"])
                if m:
                    ids.append(m.group(1))
            return sorted(set(rid for rid in ids if self._is_valid_race_id(rid)))

        return self._fetch_day_ids_future(date_str)

    def _fetch_day_ids_future(self, date_str: str) -> list[str]:
        """
        race.netkeiba.com から未来日のレースID一覧を取得する。
        1. race_list_get_date_list.html で current_group を取得
        2. race_list_sub.html でレースID一覧を取得
        """
        race_url = self.cfg.get("race_url", "https://race.netkeiba.com")

        soup = self.get(
            f"{race_url}/top/race_list_get_date_list.html?kaisai_date={date_str}&encoding=UTF-8",
            fresh_session=True,
        )
        if soup is None:
            return []

        group = None
        for li in soup.find_all("li", attrs={"date": date_str}):
            group = li.get("group")
            break
        if not group:
            return []

        soup2 = self.get(f"{race_url}/top/race_list_sub.html?kaisai_date={date_str}&current_group={group}")
        if soup2 is None:
            return []

        ids = []
        for a_tag in soup2.find_all("a", href=RACE_ID_PATTERN):
            m = RACE_ID_PATTERN.search(a_tag["href"])
            if m:
                ids.append(m.group(1))
        return sorted(set(rid for rid in ids if self._is_valid_race_id(rid)))

    def load_scraped_ids(self, path: str) -> set[str]:
        p = Path(path)
        if not p.exists():
            return set()
        return set(p.read_text().splitlines())

    def save_scraped_id(self, race_id: str, path: str):
        with open(path, "a") as f:
            f.write(race_id + "\n")
