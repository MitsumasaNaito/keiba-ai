"""
オッズページのHTML解析
"""
import logging
import re

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class OddsParser:
    def parse(self, html: str, tab_type: str) -> dict:
        soup = BeautifulSoup(html, "lxml")
        if tab_type == "win_place":
            return self._parse_win_place(soup)
        elif tab_type == "exacta":
            return self._parse_exacta(soup)
        elif tab_type == "trio":
            return self._parse_combination(soup, "trio")
        elif tab_type == "trifecta":
            return self._parse_combination(soup, "trifecta")
        return {}

    def _parse_win_place(self, soup: BeautifulSoup) -> dict:
        """単勝・複勝オッズを取得"""
        result = {"win": {}, "place": {}}
        table = soup.find("table", id="odds_tan_fuku_block")
        if table is None:
            table = soup.find("table", class_=re.compile(r"Odds_Table"))
        if table is None:
            return result

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 4:
                continue
            try:
                horse_num = int(tds[0].get_text(strip=True))
                win_odds = float(tds[2].get_text(strip=True))
                place_text = tds[3].get_text(strip=True)
                place_parts = place_text.split("-")
                if len(place_parts) == 2:
                    place_odds = (float(place_parts[0]), float(place_parts[1]))
                else:
                    place_odds = (float(place_text), float(place_text))
                result["win"][horse_num] = win_odds
                result["place"][horse_num] = place_odds
            except (ValueError, IndexError):
                continue

        return result

    def _parse_exacta(self, soup: BeautifulSoup) -> dict:
        """馬連オッズを取得"""
        result = {}
        table = soup.find("table", id="odds_umaren_block")
        if table is None:
            return result

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 3:
                continue
            try:
                combo_text = tds[0].get_text(strip=True)
                parts = re.findall(r"\d+", combo_text)
                if len(parts) >= 2:
                    key = (int(parts[0]), int(parts[1]))
                    result[key] = float(tds[2].get_text(strip=True))
            except (ValueError, IndexError):
                continue

        return result

    def _parse_combination(self, soup: BeautifulSoup, bet_type: str) -> dict:
        """3連複・3連単オッズを取得"""
        result = {}
        table_id = "odds_sanfuku_block" if bet_type == "trio" else "odds_santan_block"
        table = soup.find("table", id=table_id)
        if table is None:
            return result

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) < 2:
                continue
            try:
                combo_text = tds[0].get_text(strip=True)
                parts = tuple(int(x) for x in re.findall(r"\d+", combo_text))
                if len(parts) >= 3:
                    result[parts] = float(tds[1].get_text(strip=True))
            except (ValueError, IndexError):
                continue

        return result
