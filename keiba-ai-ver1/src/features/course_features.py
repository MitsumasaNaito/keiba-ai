"""
コース・レース条件特徴量の生成
"""
import pandas as pd

COURSE_TYPE_MAP = {"芝": 0, "ダート": 1}
DIRECTION_MAP = {"右": 0, "左": 1, "直線": 2, "": -1}
TRACK_CONDITION_MAP = {"良": 0, "稍重": 1, "重": 2, "不良": 3}
WEATHER_MAP = {"晴": 0, "曇": 1, "雨": 2, "小雨": 2, "雪": 3, "小雪": 3}

GRADE_MAP = {
    "G1": 5, "GI": 5,
    "G2": 4, "GII": 4,
    "G3": 3, "GIII": 3,
    "OP": 2, "オープン": 2,
    "L": 2,  # Listed
}


def compute_course_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    レースヘッダー情報を数値にエンコードする
    df: race_df（全レース結果が結合されたもの、またはシングルレース）
    """
    df = df.copy()

    df["course_type_enc"] = df["course_type"].map(COURSE_TYPE_MAP).fillna(-1).astype(int)
    df["direction_enc"] = df["direction"].map(DIRECTION_MAP).fillna(-1).astype(int)
    df["track_condition_enc"] = df["track_condition"].map(TRACK_CONDITION_MAP).fillna(-1).astype(int)
    df["weather_enc"] = df["weather"].map(WEATHER_MAP).fillna(-1).astype(int)
    df["venue_code_enc"] = pd.to_numeric(df["venue_code"], errors="coerce").fillna(-1).astype(int)

    # グレード
    def parse_grade(name: str) -> int:
        if pd.isna(name):
            return 1
        for key, val in GRADE_MAP.items():
            if key in str(name):
                return val
        return 1

    if "race_name" in df.columns:
        df["race_grade"] = df["race_name"].apply(parse_grade)
    else:
        df["race_grade"] = 1

    return df
