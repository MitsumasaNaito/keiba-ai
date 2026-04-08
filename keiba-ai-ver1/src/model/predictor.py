"""
モデル推論と確率変換
LightGBM のランキングスコアを各馬の勝率・複勝率に変換する
"""
import logging

import numpy as np
import pandas as pd
from scipy.special import softmax

from .model_store import ModelStore
from ..features.feature_pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, config: dict):
        self.store = ModelStore(config)
        # 本番モデルが存在すれば優先して使用する
        self.model = self.store.load(production=self.store.exists(production=True))


    def predict_race(self, race_df: pd.DataFrame) -> pd.DataFrame:
        """
        1レースの予測を実行
        戻り値: race_df に win_prob / top3_prob カラムを追加したDataFrame
        """
        df = race_df.copy()
        X = df[FEATURE_COLUMNS].astype(float)
        raw_scores = self.model.predict(X)

        # Softmax で勝率に変換
        win_probs = softmax(raw_scores)
        df["raw_score"] = raw_scores
        df["win_prob"] = win_probs

        # 複勝率の近似: Plackett-Luce モデルによる近似
        df["top3_prob"] = self._estimate_top3_prob(win_probs)

        df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)
        df["pred_rank"] = range(1, len(df) + 1)
        return df

    def _estimate_top3_prob(self, win_probs: np.ndarray) -> np.ndarray:
        """
        Plackett-Luce モデルによる複勝確率の近似計算
        P(馬i が3着以内) ≈ sum over all orderings where i appears in top 3
        近似のため、Monte Carlo シミュレーションで計算する
        """
        n = len(win_probs)
        if n <= 3:
            return np.ones(n)

        n_sim = 10000
        counts = np.zeros(n)
        for _ in range(n_sim):
            # Gumbel-max trick でサンプリング
            gumbels = np.random.gumbel(size=n)
            scores = np.log(win_probs + 1e-10) + gumbels
            order = np.argsort(-scores)
            counts[order[:3]] += 1

        return counts / n_sim

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """複数レースをまとめて予測"""
        results = []
        for race_id, group in df.groupby("race_id"):
            pred = self.predict_race(group)
            pred["race_id"] = race_id
            results.append(pred)
        return pd.concat(results, ignore_index=True)
