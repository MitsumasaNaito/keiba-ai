"""
騎手特徴量の生成
過去30日の勝率・複勝率とコース別成績

最適化: 事前にjockey_idでグループ化し、numpy searchsorted で日付検索
O(n²) → O(n log n) に改善
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "jockey_win_rate_30d": np.nan,
    "jockey_top3_rate_30d": np.nan,
    "jockey_win_rate_course": np.nan,
    "jockey_recent_rides": 0,
}


def compute_jockey_features(
    race_df: pd.DataFrame,
    all_races: pd.DataFrame,
) -> pd.DataFrame:
    race_df = race_df.copy()

    if all_races.empty or race_df.empty:
        for col, val in _DEFAULTS.items():
            race_df[col] = val
        return race_df

    # all_races を jockey_id × race_date でソートしてグループ化（一度だけ）
    all_races = all_races.copy()
    all_races["race_date"] = pd.to_datetime(all_races["race_date"])
    race_df["race_date"] = pd.to_datetime(race_df["race_date"])

    # jockey_id ごとに numpy 配列を作成（高速検索用）
    jockey_data: dict[str, dict] = {}
    for jockey_id, grp in all_races.groupby("jockey_id"):
        grp = grp.sort_values("race_date")
        jockey_data[jockey_id] = {
            "dates": grp["race_date"].values.astype("datetime64[ns]"),
            "finishes": grp["finish"].values,
            "courses": grp["course_type"].values if "course_type" in grp.columns else None,
        }

    # race_df の unique (jockey_id, race_date, course_type) だけ計算してマージ
    keys = race_df[["jockey_id", "race_date", "course_type"]].drop_duplicates().reset_index(drop=True)
    timedelta_30 = np.timedelta64(30, "D")

    rows = []
    for key in keys.itertuples(index=False):
        jockey_id = key.jockey_id
        race_date = key.race_date
        course_type = key.course_type if hasattr(key, "course_type") else ""

        if not jockey_id or jockey_id not in jockey_data or pd.isna(race_date):
            rows.append({**_DEFAULTS, "jockey_id": jockey_id, "race_date": race_date, "course_type": course_type})
            continue

        data = jockey_data[jockey_id]
        dates = data["dates"]
        finishes = data["finishes"]

        race_date_np = np.datetime64(pd.Timestamp(race_date), "ns")

        # searchsorted で O(log n) の範囲取得
        past_end = int(np.searchsorted(dates, race_date_np, side="left"))   # race_date 未満
        cutoff_np = race_date_np - timedelta_30
        recent_start = int(np.searchsorted(dates, cutoff_np, side="left"))  # 30日前以降

        recent_finishes = finishes[recent_start:past_end]
        n_recent = len(recent_finishes)

        win_rate_30d = float(np.mean(recent_finishes == 1)) if n_recent > 0 else np.nan
        top3_rate_30d = float(np.mean(recent_finishes <= 3)) if n_recent > 0 else np.nan

        # コース別勝率（過去全期間）
        past_finishes = finishes[:past_end]
        if data["courses"] is not None and len(past_finishes) > 0:
            past_courses = data["courses"][:past_end]
            course_mask = past_courses == course_type
            course_finishes = past_finishes[course_mask]
            win_rate_course = float(np.mean(course_finishes == 1)) if len(course_finishes) > 0 else np.nan
        else:
            win_rate_course = np.nan

        rows.append({
            "jockey_id": jockey_id,
            "race_date": race_date,
            "course_type": course_type,
            "jockey_win_rate_30d": win_rate_30d,
            "jockey_top3_rate_30d": top3_rate_30d,
            "jockey_win_rate_course": win_rate_course,
            "jockey_recent_rides": n_recent,
        })

    feat_df = pd.DataFrame(rows)
    result = race_df.merge(feat_df, on=["jockey_id", "race_date", "course_type"], how="left")

    # デフォルト値で欠損を補完
    for col, val in _DEFAULTS.items():
        if col not in result.columns:
            result[col] = val

    return result
