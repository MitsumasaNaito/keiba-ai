"""
スクレイパーパイプライン（エントリーポイント）
"""
import logging

from .base_scraper import load_config
from .async_pipeline import run_race_scraping_async, run_horse_scraping_async

logger = logging.getLogger(__name__)


def run_race_scraping(config: dict | None = None) -> set[str]:
    if config is None:
        config = load_config()
    return run_race_scraping_async(config)


def run_horse_scraping(horse_ids: set[str], config: dict | None = None) -> None:
    if config is None:
        config = load_config()
    run_horse_scraping_async(horse_ids, config)


def run_full_pipeline(config_path: str = "config.yaml"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    config = load_config(config_path)
    horse_ids = run_race_scraping(config)
    run_horse_scraping(horse_ids, config)


if __name__ == "__main__":
    run_full_pipeline()
