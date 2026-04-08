"""
買い目レポートの出力と履歴管理
"""
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_FILE = "data/processed/bet_history.parquet"


def _str_width(s: str) -> int:
    """全角文字を2、半角文字を1として表示幅を返す"""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _ljust_w(s: str, width: int) -> str:
    """表示幅を考慮した左寄せパディング"""
    return str(s) + " " * (width - _str_width(str(s)))


def _rjust_w(s: str, width: int) -> str:
    """表示幅を考慮した右寄せパディング"""
    return " " * (width - _str_width(str(s))) + str(s)


def _format_table(df: pd.DataFrame) -> str:
    """枠線付きテーブルを文字列で返す（全角文字対応）"""
    cols = list(df.columns)
    col_widths = [
        max(_str_width(str(c)), df[c].astype(str).map(_str_width).max())
        for c in cols
    ]

    def is_numeric(v: str) -> bool:
        return v.replace(".", "").replace("%", "").replace("-", "").replace(",", "").replace("円", "").isdigit()

    def row_str(values):
        cells = []
        for v, w in zip(values, col_widths):
            s = str(v)
            cells.append(_rjust_w(s, w) if is_numeric(s) else _ljust_w(s, w))
        return "│ " + " │ ".join(cells) + " │"

    sep_top    = "┌─" + "─┬─".join("─" * w for w in col_widths) + "─┐"
    sep_mid    = "├─" + "─┼─".join("─" * w for w in col_widths) + "─┤"
    sep_bottom = "└─" + "─┴─".join("─" * w for w in col_widths) + "─┘"

    lines = [sep_top, row_str(cols), sep_mid]
    for _, r in df.iterrows():
        lines.append(row_str(r.tolist()))
    lines.append(sep_bottom)
    return "\n".join(lines)


class BetReporter:
    def __init__(self, config: dict | None = None):  # noqa: ARG002
        self.history_path = Path(HISTORY_FILE)

    def print_report(self, race_id: str, pred_df: pd.DataFrame, bet_df: pd.DataFrame):
        """コンソールに買い目レポートを表示"""
        print(f"\n{'='*60}")
        print(f"レース: {race_id}")
        print(f"{'='*60}")

        print("\n【モデル予測ランキング（全頭）】")
        all_horses = pred_df[["horse_num", "horse_name", "win_prob", "top3_prob", "raw_score"]].copy()
        all_horses.insert(0, "順位", range(1, len(all_horses) + 1))
        all_horses["win_prob"]  = (all_horses["win_prob"]  * 100).map("{:.1f}%".format)
        all_horses["top3_prob"] = (all_horses["top3_prob"] * 100).map("{:.1f}%".format)
        all_horses["raw_score"] = all_horses["raw_score"].map("{:.4f}".format)
        all_horses = all_horses.rename(columns={
            "horse_num": "馬番", "horse_name": "馬名",
            "win_prob": "単勝確率", "top3_prob": "複勝確率", "raw_score": "スコア",
        })
        print(_format_table(all_horses))

        if bet_df.empty:
            print("\n【買い目】期待値の高い買い目なし（見送り）")
        else:
            print(f"\n【買い目】（合計: {bet_df['bet_amount'].sum():,}円）")
            display = bet_df[["bet_type", "combination", "odds", "ev", "kelly_ratio", "bet_amount"]].copy()
            display["ev"] = display["ev"].map("{:.3f}".format)
            display["kelly_ratio"] = (display["kelly_ratio"] * 100).map("{:.2f}%".format)
            display["bet_amount"] = display["bet_amount"].map("{:,}円".format)
            display = display.rename(columns={
                "bet_type": "馬券種", "combination": "組合せ",
                "odds": "オッズ", "ev": "期待値",
                "kelly_ratio": "Kelly比率", "bet_amount": "購入金額",
            })
            print(_format_table(display))
        print()

    def save_bets(self, race_id: str, bet_df: pd.DataFrame):
        """買い目をParquetに保存（後から的中確認するため）"""
        if bet_df.empty:
            return
        df = bet_df.copy()
        df["race_id"] = race_id
        df["recorded_at"] = datetime.now().isoformat()
        df["result"] = None  # 的中は後で更新

        if self.history_path.exists():
            existing = pd.read_parquet(self.history_path)
            df = pd.concat([existing, df], ignore_index=True)

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.history_path, index=False)

    def update_results(self, race_id: str, results: dict[str, bool]):
        """
        的中結果を更新する
        results: {"単勝-1": True, "馬連-1-3": False, ...}
        """
        if not self.history_path.exists():
            return
        df = pd.read_parquet(self.history_path)
        for combo_key, hit in results.items():
            parts = combo_key.split("-", 1)
            if len(parts) == 2:
                bet_type, combo = parts
                mask = (df["race_id"] == race_id) & (df["bet_type"] == bet_type) & (df["combination"] == combo)
                df.loc[mask, "result"] = hit
        df.to_parquet(self.history_path, index=False)

    def summary(self) -> pd.DataFrame:
        """回収率サマリーを表示"""
        if not self.history_path.exists():
            print("履歴データなし")
            return pd.DataFrame()

        df = pd.read_parquet(self.history_path)
        df_done = df[df["result"].notna()]

        if df_done.empty:
            print("的中結果未入力のデータのみ")
            return pd.DataFrame()

        total_bet = df_done["bet_amount"].sum()
        hits = df_done[df_done["result"]]
        total_return = (hits["odds"] * hits["bet_amount"]).sum()
        roi = total_return / total_bet if total_bet > 0 else 0

        summary = df_done.groupby("bet_type").apply(
            lambda g: pd.Series({
                "賭けた回数": len(g),
                "的中数": g["result"].sum(),
                "的中率": f"{g['result'].mean():.1%}",
                "合計賭け金": f"{g['bet_amount'].sum():,}円",
                "回収率": f"{(g[g['result']]['odds'] * g[g['result']]['bet_amount']).sum() / g['bet_amount'].sum():.1%}",
            })
        ).reset_index()

        print(f"\n全体回収率: {roi:.1%} (賭け金: {total_bet:,}円 / 回収: {total_return:,}円)")
        print(summary.to_string(index=False))
        return summary
