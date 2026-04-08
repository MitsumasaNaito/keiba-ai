"""
競馬AI メインエントリーポイント
各フェーズを CLI から実行できる

使い方:
  python main.py scrape              # データ収集（レース結果 + 馬歴）
  python main.py scrape_horses       # 馬歴のみ取得（レース結果は再取得しない）
  python main.py fix_meta            # race_date 等の欠損メタ情報を再取得して補完
  python main.py fix_meta --probe    # 1件だけHTMLを確認（クラス名調査用）
  python main.py features            # 特徴量構築
  python main.py train               # 評価モデル学習（train/test 分割あり）
  python main.py retrain             # 本番モデル再学習（全データ使用）
  python main.py list_races           # 今日のレース一覧（race_id 付き）
  python main.py list_races 2025-06-01  # 指定日のレース一覧
  python main.py predict <race_id>   # 単一レースの予測＋買い目出力
  python main.py summary             # 過去の買い目履歴サマリー
"""
import logging
import sys

import yaml
from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """tqdm.write() 経由でログを出力するハンドラ（プログレスバーと干渉しない）"""
    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        tqdm.write(msg)


def _setup_logging():
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler = TqdmLoggingHandler()
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


_setup_logging()
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_scrape(config: dict):
    from src.scraper.pipeline import run_race_scraping, run_horse_scraping
    horse_ids = run_race_scraping(config)
    run_horse_scraping(horse_ids, config)


def cmd_scrape_horses(config: dict):
    """馬歴のみ取得する（既存レース parquet から馬IDを収集して馬歴を取得）"""
    from pathlib import Path
    from src.scraper.pipeline import run_horse_scraping
    import pandas as pd

    raw_races_dir = Path(config["paths"]["raw_races"])
    parquet_files = list(raw_races_dir.glob("*.parquet"))
    if not parquet_files:
        print("レースデータがありません。先に scrape を実行してください。")
        return

    print(f"レースファイル {len(parquet_files)} 件から馬IDを収集中...")
    horse_ids: set[str] = set()
    for p in parquet_files:
        try:
            df = pd.read_parquet(p, columns=["horse_id"])
            horse_ids.update(df["horse_id"].dropna().unique())
        except Exception:
            pass
    print(f"馬ID総数: {len(horse_ids)}")
    run_horse_scraping(horse_ids, config)


def cmd_fix_meta(config: dict, probe: bool = False):
    from src.scraper.race_meta_fixer import run_fix_meta
    run_fix_meta(config, probe=probe)


def cmd_clean_races(config: dict, dry_run: bool = True):
    """キャッシュにないレースのparquetを削除する"""
    from pathlib import Path
    from src.scraper.race_id_collector import RaceIdCollector

    collector = RaceIdCollector(config)
    cfg = config["scraper"]
    all_race_ids = set(collector.collect_years(cfg["target_years"], until=cfg.get("target_until")))

    raw_races_dir = Path(config["paths"]["raw_races"])
    all_parquets = list(raw_races_dir.glob("*.parquet"))

    targets = [p for p in all_parquets if p.stem not in all_race_ids]

    print(f"parquet総数: {len(all_parquets)}")
    print(f"キャッシュ内レースID数: {len(all_race_ids)}")
    print(f"削除対象parquet: {len(targets)} 件")
    if targets:
        print(f"削除対象サンプル (先頭5件): {[p.name for p in targets[:5]]}")

    # scraped_ids.txt からキャッシュにないIDを除去
    scraped_ids_path = Path(config["paths"]["scraped_ids"])
    scraped_ids = collector.load_scraped_ids(str(scraped_ids_path))
    stale_ids = scraped_ids - all_race_ids
    print(f"\nscraped_ids.txt 総数: {len(scraped_ids)}")
    print(f"削除対象ID (scraped_ids.txt): {len(stale_ids)} 件")

    if dry_run:
        print("\n[dry-run] 実際に削除するには: python main.py clean_races --force")
        return

    for p in targets:
        p.unlink()
    print(f"{len(targets)} 件のparquetを削除しました。")

    if stale_ids:
        valid_ids = scraped_ids - stale_ids
        scraped_ids_path.write_text("\n".join(sorted(valid_ids)) + "\n")
        print(f"scraped_ids.txt から {len(stale_ids)} 件を削除しました。")


def cmd_features(config: dict):
    from src.features.feature_pipeline import build_features
    build_features(config)


def cmd_train(config: dict):
    from src.features.feature_pipeline import load_features
    from src.model.trainer import Trainer

    df = load_features(config)
    trainer = Trainer(config)
    model = trainer.train(df)

    importance = trainer.get_feature_importance(model)
    print("\n【特徴量重要度 Top20】")
    print(importance.head(20).to_string(index=False))


def cmd_retrain(config: dict):
    from src.features.feature_pipeline import load_features
    from src.model.trainer import Trainer

    df = load_features(config)
    trainer = Trainer(config)
    model = trainer.retrain(df)

    importance = trainer.get_feature_importance(model)
    print("\n【特徴量重要度 Top20（本番モデル）】")
    print(importance.head(20).to_string(index=False))


def cmd_predict(config: dict, race_id: str):
    from src.scraper.race_result_scraper import RaceResultScraper
    from src.scraper.horse_history_scraper import HorseHistoryScraper
    from src.features.horse_features import compute_horse_features
    from src.features.jockey_features import compute_jockey_features
    from src.features.course_features import compute_course_features
    from src.features.raw_loader import RawLoader
    from src.model.predictor import Predictor
    from src.odds.realtime_odds_fetcher import RealtimeOddsFetcher
    from src.betting.bet_optimizer import BetOptimizer
    from src.betting.bet_reporter import BetReporter

    logger.info(f"Predicting race: {race_id}")

    # --- 出走表取得（結果ページ → 出馬表ページの順で試みる） ---
    race_scraper = RaceResultScraper(config)
    race_df = race_scraper.scrape(race_id)
    if race_df is None or race_df.empty:
        logger.info(f"結果ページなし。出馬表ページを試みます: {race_id}")
        from src.scraper.shutuba_scraper import ShutubaScraper
        shutuba_scraper = ShutubaScraper(config)
        race_df = shutuba_scraper.scrape(race_id)
    if race_df is None or race_df.empty:
        logger.error(f"Failed to get race data for {race_id}")
        sys.exit(1)

    # --- 過去データ読み込み ---
    loader = RawLoader(config)
    all_races = loader.load_races()

    horse_ids = race_df["horse_id"].dropna().unique()
    horse_scraper = HorseHistoryScraper(config)
    horse_histories = {}
    for hid in horse_ids:
        h = horse_scraper.load_horse(hid)
        if h is None:
            logger.info(f"Fetching horse history for {hid}...")
            h = horse_scraper.scrape(hid)
            if h is not None:
                horse_scraper.save(h, hid)
        if h is not None:
            import pandas as pd
            h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce")
            horse_histories[hid] = h

    # --- 特徴量生成 ---
    import pandas as pd
    race_df["race_date"] = pd.to_datetime(race_df["race_date"], errors="coerce")
    # sex_age → sex / age の分割（出馬表取得時は未分割のため）
    if "sex_age" in race_df.columns and "age" not in race_df.columns:
        race_df["sex"] = race_df["sex_age"].str[0]
        race_df["age"] = pd.to_numeric(race_df["sex_age"].str[1:], errors="coerce")
    race_df = compute_course_features(race_df)
    race_df = compute_horse_features(race_df, horse_histories)
    race_df = compute_jockey_features(race_df, all_races)

    # field_size 補完
    if "field_size" not in race_df.columns or race_df["field_size"].isna().all():
        race_df["field_size"] = len(race_df)

    # --- 予測 ---
    predictor = Predictor(config)
    pred_df = predictor.predict_race(race_df)

    # --- リアルタイムオッズ取得 ---
    logger.info("Fetching real-time odds...")
    fetcher = RealtimeOddsFetcher(config)
    odds_data = fetcher.fetch_sync(race_id)

    # --- 買い目最適化 ---
    optimizer = BetOptimizer(config)
    bet_df = optimizer.optimize(pred_df, odds_data)

    # --- レポート出力 ---
    reporter = BetReporter(config)
    reporter.print_report(race_id, pred_df, bet_df)
    reporter.save_bets(race_id, bet_df)


VENUE_NAMES = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟",
    "05": "東京", "06": "中山", "07": "中京", "08": "京都",
    "09": "阪神", "10": "小倉",
}


def cmd_list_races(config: dict, date_str: str | None = None):
    """指定日（デフォルト: 今日）のレース一覧を表示する"""
    from datetime import date
    from src.scraper.race_id_collector import RaceIdCollector

    if date_str is None:
        date_str = date.today().strftime("%Y%m%d")
    else:
        date_str = date_str.replace("-", "")

    collector = RaceIdCollector(config)
    race_ids = collector.fetch_day_ids_listing(date_str)

    display_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    if not race_ids:
        print(f"{display_date} の開催はありません。")
        return

    print(f"{display_date} のレース一覧:\n")
    print(f"{'race_id':<14}  {'場所':<4}  {'回':>2}  {'日':>2}  R")
    print("-" * 36)
    for rid in race_ids:
        venue = VENUE_NAMES.get(rid[4:6], rid[4:6])
        kai = int(rid[6:8])
        nichi = int(rid[8:10])
        race_num = int(rid[10:12])
        print(f"{rid:<14}  {venue:<4}  {kai:>2}回  {nichi:>2}日  {race_num}R")


def cmd_summary(config: dict):
    from src.betting.bet_reporter import BetReporter
    reporter = BetReporter(config)
    reporter.summary()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    config = load_config()

    if command == "scrape":
        cmd_scrape(config)
    elif command == "scrape_horses":
        cmd_scrape_horses(config)
    elif command == "fix_meta":
        probe = "--probe" in sys.argv
        cmd_fix_meta(config, probe=probe)
    elif command == "clean_races":
        dry_run = "--force" not in sys.argv
        cmd_clean_races(config, dry_run=dry_run)
    elif command == "features":
        cmd_features(config)
    elif command == "train":
        cmd_train(config)
    elif command == "retrain":
        cmd_retrain(config)
    elif command == "list_races":
        date_arg = sys.argv[2] if len(sys.argv) >= 3 else None
        cmd_list_races(config, date_arg)
    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <race_id>")
            sys.exit(1)
        cmd_predict(config, sys.argv[2])
    elif command == "summary":
        cmd_summary(config)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
