"""
モデル評価
NDCG, AUC, 回収率シミュレーション等を計算する
"""
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def ndcg_at_k(relevance: np.ndarray, k: int = 3) -> float:
    """1レースの NDCG@k を計算"""
    relevance = np.array(relevance)
    k = min(k, len(relevance))
    order = np.argsort(-relevance)
    dcg = sum(relevance[order[i]] / np.log2(i + 2) for i in range(k))
    ideal = sorted(relevance, reverse=True)
    idcg = sum(ideal[i] / np.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(pred_df: pd.DataFrame) -> dict:
    """
    pred_df: predictor.predict_batch() の出力
    必須カラム: race_id, finish, win_prob, top3_prob
    """
    metrics = {}

    # NDCG@3 (レースごとに計算して平均)
    ndcg_scores = []
    for _, group in pred_df.groupby("race_id"):
        # 実際の着順からスコアを生成: 1着=最大, 最下位=0
        n = len(group)
        rel = np.maximum(0, n + 1 - group["finish"].values)
        # win_prob でソートした時の relevance
        order = np.argsort(-group["win_prob"].values)
        reordered_rel = rel[order]
        ndcg_scores.append(ndcg_at_k(reordered_rel, k=3))
    metrics["ndcg@3"] = float(np.mean(ndcg_scores))

    # AUC (1着予測)
    y_true = (pred_df["finish"] == 1).astype(int)
    if y_true.sum() > 0:
        metrics["auc_win"] = roc_auc_score(y_true, pred_df["win_prob"])

    # 複勝 AUC
    y_top3 = (pred_df["finish"] <= 3).astype(int)
    if y_top3.sum() > 0:
        metrics["auc_top3"] = roc_auc_score(y_top3, pred_df["top3_prob"])

    # 単勝回収率シミュレーション（人気1位の馬を常に買う場合）
    roi = _simulate_roi(pred_df)
    metrics.update(roi)

    return metrics


def _simulate_roi(pred_df: pd.DataFrame) -> dict:
    """
    モデルの予測確率上位の馬を単勝で購入した場合の回収率をシミュレート
    """
    if "odds" not in pred_df.columns:
        return {}

    # レースごとに win_prob 最大の馬を選択
    best = pred_df.loc[pred_df.groupby("race_id")["win_prob"].idxmax()]
    total_bet = len(best) * 100  # 1レース100円賭け
    total_return = (best[best["finish"] == 1]["odds"] * 100).sum()
    roi = total_return / total_bet if total_bet > 0 else 0.0

    return {"roi_top1_win": roi, "n_races": len(best)}
