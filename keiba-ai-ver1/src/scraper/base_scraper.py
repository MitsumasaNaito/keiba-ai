"""
スクレイピング基底クラス
セッション管理・レート制限・リトライを共通化
"""
import time
import random
import logging

import requests
import yaml
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BaseScraper:
    def __init__(self, config: dict):
        self.cfg = config["scraper"]
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.cfg["user_agent"]})

    def _wait(self):
        wait = random.uniform(
            self.cfg["request_interval_min"],
            self.cfg["request_interval_max"],
        )
        time.sleep(wait)

    def _new_session(self):
        """クッキーをリセットした新しいセッションを作成する"""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.cfg["user_agent"],
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })

    def get(self, url: str, fresh_session: bool = False) -> BeautifulSoup | None:
        if fresh_session:
            self._new_session()
        for attempt in range(self.cfg["retry_max"]):
            try:
                self._wait()
                resp = self.session.get(url, timeout=30)
                if resp.status_code == 200:
                    resp.encoding = resp.apparent_encoding
                    return BeautifulSoup(resp.text, "lxml")
                elif resp.status_code == 400:
                    # クッキーが原因の可能性があるためセッションをリセットして再試行
                    logger.warning(f"HTTP 400 for {url}, セッションをリセットして再試行")
                    self._new_session()
                    time.sleep(self.cfg["retry_wait"])
                elif resp.status_code in (429, 503):
                    wait = self.cfg["retry_wait"] * (2 ** attempt)
                    logger.warning(f"HTTP {resp.status_code} for {url}, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {resp.status_code} for {url}")
                    return None
            except requests.RequestException as e:
                logger.error(f"Request error ({attempt+1}/{self.cfg['retry_max']}): {e}")
                time.sleep(self.cfg["retry_wait"])
        logger.error(f"Failed to fetch after {self.cfg['retry_max']} attempts: {url}")
        return None
