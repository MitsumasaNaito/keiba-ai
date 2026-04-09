"""
並列スクレイピングパイプライン
requests + ThreadPoolExecutor で並列取得する
（aiohttp はnetkeiba のbot検出に引っかかるため使用しない）
適応型レート制限: エラーが続くと自動で減速し、成功が続くと元の速度に戻す
"""
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .base_scraper import load_config
from .race_id_collector import RaceIdCollector
from .race_result_scraper import RaceResultScraper
from .horse_history_scraper import HorseHistoryScraper

logger = logging.getLogger(__name__)


class IPBlockDetected(Exception):
    """IPブロック検知時に送出する例外"""
    pass


class AdaptiveRateLimiter:
    """
    エラー率に応じてインターバルを動的に調整するレート制限器
    - 連続エラーが error_threshold 回に達したら slowdown_factor 倍に減速
    - 成功が recovery_after 回続いたら元の速度に戻す
    - 400が ip_block_threshold 回連続したらIPブロックと判断して処理を停止
    """
    def __init__(self, cfg: dict):
        self.min_interval = cfg["request_interval_min"]
        self.max_interval = cfg["request_interval_max"]
        self.enabled = cfg.get("adaptive_rate_limit", True)
        self.error_threshold = cfg.get("error_threshold", 3)
        self.slowdown_factor = cfg.get("slowdown_factor", 2.0)
        self.recovery_after = cfg.get("recovery_after", 20)
        self.ip_block_threshold = cfg.get("ip_block_threshold", 20)

        self._current_factor = 1.0
        self._consecutive_errors = 0
        self._consecutive_successes = 0
        self._consecutive_400 = 0
        self._lock = threading.Lock()

    def wait(self):
        interval = random.uniform(
            self.min_interval * self._current_factor,
            self.max_interval * self._current_factor,
        )
        time.sleep(interval)

    def on_success(self):
        if not self.enabled or self._current_factor == 1.0:
            return
        with self._lock:
            self._consecutive_errors = 0
            self._consecutive_400 = 0
            self._consecutive_successes += 1
            if self._consecutive_successes >= self.recovery_after:
                self._current_factor = 1.0
                self._consecutive_successes = 0
                logger.info("レート制限: 通常速度に復帰")

    def on_error(self):
        if not self.enabled:
            return
        with self._lock:
            self._consecutive_successes = 0
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.error_threshold:
                self._current_factor = min(self._current_factor * self.slowdown_factor, 8.0)
                self._consecutive_errors = 0
                logger.warning(
                    f"レート制限: エラー連続 {self.error_threshold} 回 → "
                    f"インターバルを {self._current_factor:.1f}x に減速"
                )

    def on_400(self):
        """400連続でIPブロックを検知したら例外を送出"""
        with self._lock:
            self._consecutive_400 += 1
            if self._consecutive_400 >= self.ip_block_threshold:
                raise IPBlockDetected(
                    f"400エラーが {self.ip_block_threshold} 回連続 → IPブロックの可能性。"
                    "数時間待ってから再試行してください。"
                )

    def reset_400(self):
        with self._lock:
            self._consecutive_400 = 0

    @property
    def current_interval_range(self) -> str:
        lo = self.min_interval * self._current_factor
        hi = self.max_interval * self._current_factor
        return f"{lo:.1f}〜{hi:.1f}秒"


def _make_session(cfg: dict) -> requests.Session:
    """スレッドごとの独立したセッションを作成する"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": cfg["user_agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    })
    return session


def _fetch_html(
    url: str,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
    session: requests.Session,
) -> str | None:
    rate_limiter.wait()
    for attempt in range(cfg["retry_max"]):
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200:
                rate_limiter.on_success()
                rate_limiter.reset_400()
                resp.encoding = resp.apparent_encoding
                return resp.text
            elif resp.status_code == 400:
                rate_limiter.on_400()  # 連続400でIPBlockDetected を送出
                # 存在しないレースID等 → セッションリセットして1回だけ再試行
                session.cookies.clear()
                if attempt == 0:
                    time.sleep(cfg["retry_wait"])
                    continue
                return None
            elif resp.status_code in (429, 503):
                rate_limiter.on_error()
                wait = cfg["retry_wait"] * (2 ** attempt)
                logger.warning(f"HTTP {resp.status_code} ({url}), {wait}秒待機")
                time.sleep(wait)
            else:
                rate_limiter.on_error()
                logger.error(f"HTTP {resp.status_code} ({url})")
                return None
        except requests.RequestException as e:
            rate_limiter.on_error()
            logger.error(f"Fetch error ({attempt+1}/{cfg['retry_max']}): {e}")
            time.sleep(cfg["retry_wait"])
    return None


def _scrape_race_worker(
    race_id: str,
    scraper: RaceResultScraper,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
    scraped_ids_path: str,
    horse_ids: set,
    horse_ids_lock: threading.Lock,
    counters: dict,
    counters_lock: threading.Lock,
    file_lock: threading.Lock,
) -> None:
    url = f"{cfg['base_url']}/race/{race_id}/"
    session = _make_session(cfg)
    html = _fetch_html(url, cfg, rate_limiter, session)

    if html is None:
        with counters_lock:
            counters["fetch_fail"] += 1
        return

    soup = BeautifulSoup(html, "lxml")
    df = scraper.parse(soup, race_id)
    if df is not None:
        scraper.save(df, race_id)
        # スレッドセーフなファイル書き込み
        with file_lock:
            with open(scraped_ids_path, "a") as f:
                f.write(race_id + "\n")
        if "horse_id" in df.columns:
            with horse_ids_lock:
                horse_ids.update(df["horse_id"].dropna().unique())
        with counters_lock:
            counters["saved"] += 1
    else:
        with counters_lock:
            counters["parse_fail"] += 1
        logger.warning(f"Parse失敗（保存スキップ）: {race_id}")


def _scrape_horse_worker(
    horse_id: str,
    scraper: HorseHistoryScraper,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
) -> None:
    url = f"{cfg['base_url']}/horse/result/{horse_id}/"
    session = _make_session(cfg)
    html = _fetch_html(url, cfg, rate_limiter, session)
    if html is None:
        return

    soup = BeautifulSoup(html, "lxml")
    df = scraper.parse(soup, horse_id)
    if df is not None:
        scraper.save(df, horse_id)


def run_race_scraping_async(config: dict) -> set[str]:
    """レース結果を並列取得する（pipeline.py から呼ばれる）"""
    cfg = config["scraper"]
    concurrent = cfg.get("concurrent_requests", 8)
    rate_limiter = AdaptiveRateLimiter(cfg)
    scraped_ids_path = config["paths"]["scraped_ids"]

    collector = RaceIdCollector(config)
    scraper = RaceResultScraper(config)

    years = cfg["target_years"]
    until = cfg.get("target_until")
    logger.info(f"レースID収集中... years={years}, until={until}")
    all_race_ids = collector.collect_years(years, until=until)
    logger.info(f"レースID総数: {len(all_race_ids)}")

    # 取得済み判定: parquetファイルの存在を正とし、scraped_ids.txt で補完
    raw_races_dir = Path(config["paths"]["raw_races"])
    scraped_from_parquet = {p.stem for p in raw_races_dir.glob("*.parquet")}
    scraped_from_txt = collector.load_scraped_ids(scraped_ids_path)
    scraped = scraped_from_parquet | scraped_from_txt
    pending = [rid for rid in all_race_ids if rid not in scraped]
    already_done = len(all_race_ids) - len(pending)
    logger.info(
        f"取得済み: {already_done} / 未取得: {len(pending)} / キャッシュ内合計: {len(all_race_ids)} "
        f"(parquetファイル総数: {len(scraped_from_parquet)}, インターバル: {rate_limiter.current_interval_range}, 並列数: {concurrent})"
    )

    horse_ids: set[str] = set()
    horse_ids_lock = threading.Lock()
    counters = {"saved": 0, "fetch_fail": 0, "parse_fail": 0}
    counters_lock = threading.Lock()
    file_lock = threading.Lock()

    if pending:
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {
                executor.submit(
                    _scrape_race_worker,
                    race_id, scraper, cfg, rate_limiter,
                    scraped_ids_path, horse_ids, horse_ids_lock,
                    counters, counters_lock, file_lock,
                ): race_id
                for race_id in pending
            }
            try:
                with tqdm(total=len(pending), desc=f"レース結果取得 (並列数={concurrent})", unit="レース", dynamic_ncols=True) as pbar:
                    for future in as_completed(futures):
                        future.result()  # IPBlockDetected を伝播させる
                        pbar.update(1)
            except IPBlockDetected as e:
                logger.error(f"[IPブロック検知] {e}")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    logger.info(
        f"レース結果取得完了。保存: {counters['saved']} / "
        f"フェッチ失敗: {counters['fetch_fail']} / "
        f"パース失敗: {counters['parse_fail']} / "
        f"新規馬ID数: {len(horse_ids)}"
    )

    # 既存の parquet からも馬IDを補完（前回までのスクレイプ分）
    existing_files = list(raw_races_dir.glob("*.parquet"))
    if existing_files:
        logger.info(f"既存レースファイル {len(existing_files)} 件から馬IDを補完中...")
        for p in existing_files:
            try:
                df = pd.read_parquet(p, columns=["horse_id"])
                horse_ids.update(df["horse_id"].dropna().unique())
            except Exception:
                pass
        logger.info(f"馬ID総数（補完後）: {len(horse_ids)}")

    return horse_ids


def run_horse_scraping_async(horse_ids: set[str], config: dict) -> None:
    """馬歴を並列取得する（pipeline.py から呼ばれる）"""
    cfg = config["scraper"]
    concurrent = cfg.get("concurrent_requests", 8)
    rate_limiter = AdaptiveRateLimiter(cfg)

    scraper = HorseHistoryScraper(config)
    out_dir = Path(config["paths"]["raw_horses"])

    scraped_horse_ids = {p.stem for p in out_dir.glob("*.parquet")}
    pending = [hid for hid in horse_ids if hid and hid not in scraped_horse_ids]
    logger.info(
        f"未取得馬数: {len(pending)} / 取得済み: {len(scraped_horse_ids)} "
        f"(インターバル: {rate_limiter.current_interval_range}, 並列数: {concurrent})"
    )

    if pending:
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = {
                executor.submit(_scrape_horse_worker, horse_id, scraper, cfg, rate_limiter): horse_id
                for horse_id in pending
            }
            with tqdm(total=len(pending), desc=f"馬歴取得 (並列数={concurrent})", unit="頭", dynamic_ncols=True) as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

    logger.info("馬歴取得完了。")
