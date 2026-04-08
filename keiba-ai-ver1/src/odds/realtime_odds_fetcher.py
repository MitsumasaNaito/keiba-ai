"""
リアルタイムオッズ取得
Playwright を使用して netkeiba.com からオッズを動的取得する
"""
import asyncio
import logging
import time

from .browser_manager import BrowserManager
from .odds_parser import OddsParser

logger = logging.getLogger(__name__)

ODDS_BASE_URL = "https://race.netkeiba.com/odds/index.html"
SHUTUBA_BASE_URL = "https://race.netkeiba.com/race/shutuba.html"


class RealtimeOddsFetcher:
    def __init__(self, config: dict, headless: bool = True):
        self.cfg = config["scraper"]
        self.manager = BrowserManager(headless=headless)
        self.parser = OddsParser()

    async def fetch_race_odds(self, race_id: str) -> dict:
        """
        指定レースの各馬券種別オッズを取得する
        戻り値: {
            "win": {horse_num: odds, ...},
            "place": {horse_num: (min_odds, max_odds), ...},
            "exacta": {(horse_num1, horse_num2): odds, ...},
            ...
        }
        """
        async with self.manager.session():
            page = await self.manager.new_page()
            result = {}

            # 単勝・複勝
            url = f"{ODDS_BASE_URL}?race_id={race_id}&type=b1"
            result.update(await self._fetch_tab(page, url, "win_place"))

            await asyncio.sleep(2.0)

            # 馬連
            url = f"{ODDS_BASE_URL}?race_id={race_id}&type=b4"
            result["exacta"] = await self._fetch_tab(page, url, "exacta")

            await asyncio.sleep(2.0)

            # 3連複
            url = f"{ODDS_BASE_URL}?race_id={race_id}&type=b7"
            result["trio"] = await self._fetch_tab(page, url, "trio")

            await asyncio.sleep(2.0)

            # 3連単
            url = f"{ODDS_BASE_URL}?race_id={race_id}&type=b8"
            result["trifecta"] = await self._fetch_tab(page, url, "trifecta")

            return result

    async def _fetch_tab(self, page, url: str, tab_type: str) -> dict:
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            # オッズテーブルの表示を待つ
            await page.wait_for_selector("table.Odds_Table, .OddsTable", timeout=15000)
            html = await page.content()
            return self.parser.parse(html, tab_type)
        except Exception as e:
            logger.error(f"Failed to fetch {tab_type} odds from {url}: {e}")
            return {}

    def fetch_sync(self, race_id: str) -> dict:
        """同期インターフェース（asyncio.run() でラップ）"""
        return asyncio.run(self.fetch_race_odds(race_id))
