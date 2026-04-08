"""
買い目の最適化
Kelly Criterion を全馬・全馬券種に適用して最適な買い目リストを生成する
"""
import logging
from itertools import combinations

import numpy as np
import pandas as pd

from .kelly import kelly_fraction, kelly_top3, kelly_exacta, expected_value

logger = logging.getLogger(__name__)


class BetOptimizer:
    def __init__(self, config: dict):
        bet_cfg = config["betting"]
        self.kelly_scale = bet_cfg["kelly_fraction"]
        self.min_kelly = bet_cfg["min_kelly_threshold"]
        self.min_ev = bet_cfg["min_expected_value"]
        self.max_bet_ratio = bet_cfg["max_bet_ratio_per_race"]
        self.bankroll = bet_cfg["bankroll"]

    def optimize(
        self,
        pred_df: pd.DataFrame,
        odds_data: dict,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        pred_df : pd.DataFrame
            predictor.predict_race() の出力（win_prob, top3_prob カラムを含む）
        odds_data : dict
            realtime_odds_fetcher.fetch_race_odds() の出力

        Returns
        -------
        pd.DataFrame
            買い目・金額・期待値・Kelly比率のリスト
        """
        bets = []

        win_odds = odds_data.get("win", {})
        place_odds = odds_data.get("place", {})
        exacta_odds = odds_data.get("exacta", {})
        trio_odds = odds_data.get("trio", {})

        for _, row in pred_df.iterrows():
            horse_num = row["horse_num"]
            win_prob = row["win_prob"]
            top3_prob = row["top3_prob"]

            # 単勝
            if horse_num in win_odds:
                odds = win_odds[horse_num]
                f = kelly_fraction(win_prob, odds, self.kelly_scale)
                ev = expected_value(win_prob, odds)
                if f >= self.min_kelly and ev >= self.min_ev:
                    bets.append(self._make_bet("単勝", str(horse_num), f, ev, odds))

            # 複勝（オッズは最小値を使用）
            if horse_num in place_odds:
                min_odds, max_odds = place_odds[horse_num]
                # 保守的に最小オッズで評価
                f = kelly_top3(top3_prob, min_odds, self.kelly_scale)
                ev = expected_value(top3_prob, min_odds)
                if f >= self.min_kelly and ev >= self.min_ev:
                    bets.append(self._make_bet("複勝", str(horse_num), f, ev, min_odds))

        # 馬連
        for (h1, h2), odds in exacta_odds.items():
            p1 = self._get_prob(pred_df, h1, "win_prob")
            p2 = self._get_prob(pred_df, h2, "win_prob")
            if p1 is None or p2 is None:
                continue
            # 馬連確率の近似: P(1着) × P(2着 | 1着以外)
            p_ab = (p1 * p2 / (1 - p1 + 1e-10)) + (p2 * p1 / (1 - p2 + 1e-10))
            p_ab = min(p_ab, 0.99)
            f = kelly_exacta(p_ab, odds, self.kelly_scale)
            ev = expected_value(p_ab, odds)
            if f >= self.min_kelly and ev >= self.min_ev:
                bets.append(self._make_bet("馬連", f"{h1}-{h2}", f, ev, odds))

        # 3連複
        for combo, odds in trio_odds.items():
            if len(combo) != 3:
                continue
            probs = [self._get_prob(pred_df, h, "top3_prob") for h in combo]
            if any(p is None for p in probs):
                continue
            # 3頭全員が3着以内に入る確率の近似
            p_trio = float(np.prod(probs)) ** (1 / 2)  # 相関を考慮した近似
            f = kelly_fraction(p_trio, odds, self.kelly_scale)
            ev = expected_value(p_trio, odds)
            if f >= self.min_kelly and ev >= self.min_ev:
                combo_str = "-".join(str(h) for h in combo)
                bets.append(self._make_bet("3連複", combo_str, f, ev, odds))

        if not bets:
            return pd.DataFrame(columns=["bet_type", "combination", "kelly_ratio", "ev", "odds", "bet_amount"])

        bet_df = pd.DataFrame(bets).sort_values("ev", ascending=False)

        # 1レースの合計賭け比率に上限を適用
        total_ratio = bet_df["kelly_ratio"].sum()
        if total_ratio > self.max_bet_ratio:
            scale = self.max_bet_ratio / total_ratio
            bet_df["kelly_ratio"] = bet_df["kelly_ratio"] * scale

        # 金額計算（100円単位に切り捨て）
        bet_df["bet_amount"] = (bet_df["kelly_ratio"] * self.bankroll / 100).astype(int) * 100
        bet_df = bet_df[bet_df["bet_amount"] >= 100]

        return bet_df.reset_index(drop=True)

    def _make_bet(self, bet_type: str, combination: str, f: float, ev: float, odds: float) -> dict:
        return {
            "bet_type": bet_type,
            "combination": combination,
            "kelly_ratio": f,
            "ev": ev,
            "odds": odds,
        }

    def _get_prob(self, pred_df: pd.DataFrame, horse_num: int, prob_col: str) -> float | None:
        row = pred_df[pred_df["horse_num"] == horse_num]
        if row.empty:
            return None
        return float(row[prob_col].iloc[0])
