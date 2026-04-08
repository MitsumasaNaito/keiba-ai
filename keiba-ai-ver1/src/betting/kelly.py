"""
Kelly Criterion による賭け比率の計算
"""
import numpy as np


def kelly_fraction(
    win_prob: float,
    odds: float,
    kelly_fraction_scale: float = 0.25,
) -> float:
    """
    Kelly Criterion による最適賭け比率を計算する

    Parameters
    ----------
    win_prob : float
        モデルが推定した勝率 (0 < p < 1)
    odds : float
        単勝オッズ（払戻倍率, 例: 3.5 → 3.5倍）
    kelly_fraction_scale : float
        Fractional Kelly の係数（デフォルト: 0.25 = Quarter Kelly）

    Returns
    -------
    float
        資金に対する賭け比率（0以下の場合は見送り）
    """
    if win_prob <= 0 or win_prob >= 1 or odds <= 1:
        return 0.0

    # ネットオッズ b = 払戻倍率 - 1
    b = odds - 1.0
    q = 1.0 - win_prob

    # Kelly式: f* = (b*p - q) / b
    f_star = (b * win_prob - q) / b
    return float(f_star * kelly_fraction_scale)


def expected_value(win_prob: float, odds: float) -> float:
    """
    期待値を計算する
    E = p * odds (1以上なら期待値プラス)
    """
    return win_prob * odds


def kelly_top3(
    top3_prob: float,
    place_odds: float,
    kelly_fraction_scale: float = 0.25,
) -> float:
    """
    複勝用 Kelly Criterion
    """
    return kelly_fraction(top3_prob, place_odds, kelly_fraction_scale)


def kelly_exacta(
    p_ab: float,
    exacta_odds: float,
    kelly_fraction_scale: float = 0.25,
) -> float:
    """
    馬連用 Kelly Criterion
    p_ab: 馬 A と B が1・2着に入る確率の推定値
    """
    return kelly_fraction(p_ab, exacta_odds, kelly_fraction_scale)
