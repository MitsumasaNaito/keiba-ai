"""
特徴量パイプライン
全特徴量を結合してトレーニング用・予測用データを生成する
"""
import logging
from pathlib import Path

import pandas as pd

from .raw_loader import RawLoader
from .horse_features import compute_horse_features
from .jockey_features import compute_jockey_features
from .course_features import compute_course_features

logger = logging.getLogger(__name__)

# LightGBM に渡す特徴量カラム
FEATURE_COLUMNS = [
    # コース・レース条件
    "course_type_enc",
    "direction_enc",
    "distance",
    "track_condition_enc",
    "weather_enc",
    "venue_code_enc",
    "race_grade",
    "field_size",
    "gate",
    "horse_num",
    # 斤量
    "weight_carried",
    # 馬
    "age",
    "horse_win_rate_all",
    "horse_top3_rate_all",
    "horse_top3_rate_3",
    "horse_last_finish",
    "horse_days_since_last",
    "horse_avg_last3f_3",
    "horse_career_races",
    "horse_running_style",
    "horse_weight",
    "weight_diff",
    # 騎手
    "jockey_win_rate_30d",
    "jockey_top3_rate_30d",
    "jockey_win_rate_course",
    "jockey_recent_rides",
]

TARGET_COLUMN = "finish"


def build_features(config: dict) -> pd.DataFrame:
    """
    生データから特徴量データを構築して Parquet に保存する
    """
    loader = RawLoader(config)

    logger.info("Loading raw race data...")
    races = loader.load_races()
    if races.empty:
        raise ValueError("No race data found. Run scraping first.")
    logger.info(f"Loaded {len(races)} race entries from {races['race_id'].nunique()} races")

    logger.info("Loading horse histories...")
    horse_ids = races["horse_id"].dropna().unique()
    horse_histories = {}
    for hid in horse_ids:
        h = loader.load_horse(hid)
        if h is not None:
            h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce")
            horse_histories[hid] = h
    logger.info(f"Loaded histories for {len(horse_histories)} horses")

    # race_date が空の場合は race_id 先頭4桁から年だけ補完（月日は1/1で代替）
    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    missing_date = races["race_date"].isna()
    if missing_date.any():
        approx = pd.to_datetime(races.loc[missing_date, "race_id"].str[:4] + "-01-01", errors="coerce")
        races.loc[missing_date, "race_date"] = approx
        logger.warning(f"race_date が空の行 {missing_date.sum()} 件を race_id から年のみ補完しました")

    logger.info("Computing course features...")
    races = compute_course_features(races)

    logger.info("Computing horse features...")
    races = compute_horse_features(races, horse_histories)

    logger.info("Computing jockey features...")
    races = compute_jockey_features(races, races)

    # フィールドサイズを各レースの出走頭数で補完
    if "field_size" not in races.columns:
        races["field_size"] = races.groupby("race_id")["horse_num"].transform("max")

    # ターゲット変数
    races["target"] = races[TARGET_COLUMN]
    # lambdarank 用スコア: 1着=18点, 18着=0点 (最大頭数を18と仮定)
    races["rank_score"] = (19 - races["finish"]).clip(lower=0)

    out_path = Path(config["paths"]["processed"]) / "features_train.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    races.to_parquet(out_path, index=False)
    logger.info(f"Saved features to {out_path}")
    return races


def load_features(config: dict) -> pd.DataFrame:
    path = Path(config["paths"]["processed"]) / "features_train.parquet"
    return pd.read_parquet(path)
