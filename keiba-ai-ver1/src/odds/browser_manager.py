"""
Playwright ブラウザ管理
リアルタイムオッズ取得のためのブラウザセッション管理
"""
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class BrowserManager:
    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser = None
        self._playwright = None

    async def start(self):
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        logger.info("Browser started")

    async def stop(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        logger.info("Browser stopped")

    async def new_page(self):
        if self._browser is None:
            raise RuntimeError("Browser not started. Call start() first.")
        page = await self._browser.new_page()
        await page.set_extra_http_headers({
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        })
        return page

    @asynccontextmanager
    async def session(self):
        await self.start()
        try:
            yield self
        finally:
            await self.stop()
