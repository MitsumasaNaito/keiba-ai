"""
馬固有の特徴量生成
直近成績・コース適性・通算成績などを計算する
未来情報リーク防止のため、対象レース日より前のデータのみ使用する
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def time_to_seconds(t: str) -> float | None:
    """タイム文字列（例: "1:34.5"）を秒数に変換"""
    try:
        parts = str(t).split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, TypeError):
        return None


def compute_horse_features(
    race_df: pd.DataFrame,
    horse_histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    race_df: 対象レースの出走馬テーブル（1行=1頭）
    horse_histories: {horse_id: 過去成績DataFrame} の辞書
    戻り値: race_df に特徴量カラムを追加したDataFrame
    """
    feature_rows = []
    for _, row in race_df.iterrows():
        horse_id = row.get("horse_id", "")
        race_date = row.get("race_date")
        feats = _compute_single_horse(horse_id, race_date, horse_histories)
        feature_rows.append(feats)

    feat_df = pd.DataFrame(feature_rows, index=race_df.index)
    return pd.concat([race_df, feat_df], axis=1)


def _compute_single_horse(
    horse_id: str,
    race_date,
    horse_histories: dict[str, pd.DataFrame],
) -> dict:
    prefix = "horse_"
    defaults = {
        f"{prefix}win_rate_all": np.nan,
        f"{prefix}win_rate_course": np.nan,
        f"{prefix}top3_rate_all": np.nan,
        f"{prefix}top3_rate_3": np.nan,
        f"{prefix}last_finish": np.nan,
        f"{prefix}days_since_last": np.nan,
        f"{prefix}avg_last3f_3": np.nan,
        f"{prefix}career_races": 0,
        f"{prefix}running_style": np.nan,
    }

    hist = horse_histories.get(horse_id)
    if hist is None or hist.empty:
        return defaults

    # 未来情報リーク防止: 対象レース日より前のみ使用
    if race_date is not None and "race_date" in hist.columns:
        hist = hist.copy()
        hist["race_date"] = pd.to_datetime(hist["race_date"], errors="coerce")
        hist = hist[hist["race_date"] < pd.Timestamp(race_date)]

    if hist.empty:
        return defaults

    hist = hist.sort_values("race_date", ascending=False)
    career = len(hist)
    wins = (hist["finish"] == 1).sum()
    top3 = (hist["finish"] <= 3).sum()

    last_finish = hist["finish"].iloc[0]
    days_since = None
    if "race_date" in hist.columns and race_date is not None:
        last_date = hist["race_date"].iloc[0]
        if pd.notna(last_date):
            days_since = (pd.Timestamp(race_date) - last_date).days

    # 直近3走複勝率
    recent3 = hist.head(3)
    top3_rate_3 = (recent3["finish"] <= 3).mean() if len(recent3) > 0 else np.nan

    # 上がり3F平均（直近3走）
    avg_last3f = np.nan
    if "last3f" in hist.columns:
        recent3f = hist["last3f"].head(3).dropna()
        if len(recent3f) > 0:
            avg_last3f = recent3f.mean()

    # 脚質推定（平均コーナー4角位置 vs 出走頭数）
    running_style = _estimate_running_style(hist)

    return {
        f"{prefix}win_rate_all": wins / career if career > 0 else 0,
        f"{prefix}top3_rate_all": top3 / career if career > 0 else 0,
        f"{prefix}top3_rate_3": top3_rate_3,
        f"{prefix}last_finish": last_finish,
        f"{prefix}days_since_last": days_since,
        f"{prefix}avg_last3f_3": avg_last3f,
        f"{prefix}career_races": career,
        f"{prefix}running_style": running_style,
    }


def _estimate_running_style(hist: pd.DataFrame) -> float:
    """
    コーナー通過順位（4角位置/頭数）の平均から脚質スコアを推定
    0=逃げ, 1=追込 に近い連続値を返す
    """
    if "corner_positions" not in hist.columns or "field_size" not in hist.columns:
        return np.nan

    ratios = []
    for _, row in hist.head(5).iterrows():
        corners = str(row.get("corner_positions", ""))
        positions = [int(x) for x in corners.replace("-", ",").split(",") if x.strip().isdigit()]
        field = row.get("field_size")
        if positions and field and field > 0:
            pos4 = positions[-1]
            ratios.append(pos4 / field)

    return float(np.mean(ratios)) if ratios else np.nan
