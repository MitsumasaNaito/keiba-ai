"""
レースメタ情報の差分補完
race_date / race_name / weather / track_condition が空の parquet を再取得して上書きする

使い方:
  python main.py fix_meta           # 空メタの全レースを修正
  python main.py fix_meta --probe   # 1件だけ取得してHTMLを標準出力（クラス名確認用）
"""
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

from .async_pipeline import AdaptiveRateLimiter, IPBlockDetected, _make_session, _fetch_html
from .race_result_scraper import RaceResultScraper

logger = logging.getLogger(__name__)

# メタ情報として更新するカラム
META_COLUMNS = ["race_date", "race_name", "weather", "track_condition", "direction", "venue_code"]


def _needs_fix(parquet_path: Path) -> bool:
    """race_date が空なら修正が必要"""
    try:
        df = pd.read_parquet(parquet_path, columns=["race_date"])
        val = df["race_date"].iloc[0] if len(df) > 0 else ""
        return str(val).strip() == "" or pd.isna(val)
    except Exception:
        return False


def _fix_race_worker(
    race_id: str,
    parquet_path: Path,
    scraper: RaceResultScraper,
    cfg: dict,
    rate_limiter: AdaptiveRateLimiter,
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
    meta = scraper._parse_race_meta(soup, race_id)
    if meta is None:
        with counters_lock:
            counters["parse_fail"] += 1
        logger.warning(f"メタ解析失敗: {race_id}")
        return

    # 既存 parquet を読み込んでメタカラムだけ上書き
    try:
        with file_lock:
            df = pd.read_parquet(parquet_path)
            for col in META_COLUMNS:
                if col in meta:
                    df[col] = meta[col]
            df.to_parquet(parquet_path, index=False)
        with counters_lock:
            counters["fixed"] += 1
    except Exception as e:
        logger.error(f"保存失敗 {race_id}: {e}")
        with counters_lock:
            counters["save_fail"] += 1


def run_fix_meta(config: dict, probe: bool = False) -> None:
    """
    メタ情報が欠損しているレースを再スクレイプして補完する

    probe=True の場合は最初の1件だけ取得してHTMLを標準出力し終了
    （正しいCSSクラス名の確認用）
    """
    cfg = config["scraper"]
    raw_races_dir = Path(config["paths"]["raw_races"])

    all_parquets = sorted(raw_races_dir.glob("*.parquet"))
    logger.info(f"parquetファイル総数: {len(all_parquets)}")

    pending_paths = [p for p in all_parquets if _needs_fix(p)]
    logger.info(f"メタ情報が空のレース: {len(pending_paths)} 件")

    if not pending_paths:
        logger.info("修正が必要なレースはありません。")
        return

    if probe:
        _probe_html(pending_paths[0], cfg)
        return

    concurrent = cfg.get("concurrent_requests", 3)
    rate_limiter = AdaptiveRateLimiter(cfg)
    scraper = RaceResultScraper(config)

    counters = {"fixed": 0, "fetch_fail": 0, "parse_fail": 0, "save_fail": 0}
    counters_lock = threading.Lock()
    file_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = {
            executor.submit(
                _fix_race_worker,
                p.stem, p, scraper, cfg, rate_limiter,
                counters, counters_lock, file_lock,
            ): p.stem
            for p in pending_paths
        }
        try:
            with tqdm(total=len(pending_paths), desc=f"メタ補完 (並列数={concurrent})", unit="レース") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)
        except IPBlockDetected as e:
            logger.error(f"[IPブロック検知] {e}")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    logger.info(
        f"完了: 修正={counters['fixed']} / フェッチ失敗={counters['fetch_fail']} / "
        f"パース失敗={counters['parse_fail']} / 保存失敗={counters['save_fail']}"
    )

    # features を再生成するよう案内
    if counters["fixed"] > 0:
        logger.info("メタ補完が完了しました。'python main.py features' を再実行してください。")


def _probe_html(parquet_path: Path, cfg: dict) -> None:
    """
    1件だけHTMLを取得して構造を標準出力する
    正しいCSSクラス名の確認に使う
    """
    race_id = parquet_path.stem
    url = f"{cfg['base_url']}/race/{race_id}/"
    logger.info(f"プローブ取得: {url}")

    rate_limiter = AdaptiveRateLimiter(cfg)
    session = _make_session(cfg)
    html = _fetch_html(url, cfg, rate_limiter, session)

    if html is None:
        logger.error("取得失敗（IPブロック中の可能性）")
        return

    soup = BeautifulSoup(html, "lxml")

    print("\n" + "="*60)
    print(f"[PROBE] race_id: {race_id}")
    print("="*60)

    # タイトル
    title = soup.find("title")
    print(f"\n<title>: {title.get_text(strip=True) if title else 'なし'}")

    # h1 タグ一覧
    print("\n--- h1 タグ ---")
    for tag in soup.find_all("h1"):
        print(f"  class={tag.get('class')} text={tag.get_text(strip=True)[:50]}")

    # div でよくありそうなクラスを確認
    print("\n--- race関連 div/p クラス ---")
    for tag in soup.find_all(["div", "p", "ul", "li"]):
        cls = tag.get("class")
        text = tag.get_text(strip=True)[:60]
        if cls and any(kw in " ".join(cls).lower() for kw in ["race", "data", "info", "date", "place", "head"]):
            print(f"  <{tag.name} class={cls}> {text}")

    print("\n--- HTML先頭3000文字 ---")
    print(html[:3000])
