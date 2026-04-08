# 競馬AI

netkeiba.com から過去レースデータを収集し、LightGBM で着順を予測、Kelly Criterion で最適な買い目を算出する競馬予測システム。

## プロジェクト構成

```
keiba-ai-ver1/
├── config.yaml                   # 全設定（URL・パス・モデルパラメータ・Kelly設定）
├── main.py                       # CLI エントリーポイント
├── pyproject.toml
├── src/
│   ├── scraper/
│   │   ├── base_scraper.py       # レート制限・リトライ・セッション管理
│   │   ├── race_id_collector.py  # 月別レースID一覧収集
│   │   ├── race_result_scraper.py  # レース結果取得 → Parquet 保存
│   │   ├── shutuba_scraper.py    # 出馬表取得（開催前レース向け）
│   │   ├── horse_history_scraper.py  # 馬の過去成績取得
│   │   ├── async_pipeline.py     # 並列スクレイピング（aiohttp）
│   │   └── pipeline.py           # エントリーポイント（asyncio.run でラップ）
│   ├── features/
│   │   ├── raw_loader.py         # Parquet ロード
│   │   ├── horse_features.py     # 馬特徴量（勝率・複勝率・脚質推定）
│   │   ├── jockey_features.py    # 騎手特徴量（直近30日成績）
│   │   ├── course_features.py    # コース条件のエンコーディング
│   │   └── feature_pipeline.py   # 全特徴量結合（27特徴量）
│   ├── model/
│   │   ├── trainer.py            # LightGBM 学習（評価モデル・本番モデル）
│   │   ├── predictor.py          # 推論 + 確率変換
│   │   ├── evaluator.py          # NDCG・AUC・回収率シミュレーション
│   │   └── model_store.py        # モデル保存・ロード
│   ├── odds/
│   │   ├── browser_manager.py    # Playwright ブラウザ管理
│   │   ├── realtime_odds_fetcher.py  # リアルタイムオッズ取得
│   │   └── odds_parser.py        # HTML 解析 → 辞書変換
│   └── betting/
│       ├── kelly.py              # Kelly Criterion 計算
│       ├── bet_optimizer.py      # 全馬券種の買い目最適化
│       └── bet_reporter.py       # 買い目表示・履歴記録・回収率集計
├── data/
│   ├── raw/
│   │   ├── races/                # レース結果（race_id.parquet）
│   │   └── horses/               # 馬別過去成績（horse_id.parquet）
│   └── processed/
│       ├── features_train.parquet
│       ├── bet_history.parquet
│       └── models/
│           ├── lgb_model.txt             # 評価モデル
│           └── lgb_model_production.txt  # 本番モデル
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_feature_analysis.ipynb
    └── 03_model_evaluation.ipynb
```

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

開発用ツール（Jupyter）も含める場合：

```bash
uv sync --dev
```

### 2. Playwright ブラウザのインストール

リアルタイムオッズ取得に使用する Chromium をインストールする。

```bash
uv run playwright install chromium
```

---

## 使い方

### ステップ 1: データ収集

`config.yaml` の `target_years` / `target_until` に指定した期間のレース結果と馬の過去成績を取得する。
`aiohttp` による並列取得（デフォルト同時5接続）のため、逐次取得と比べて約5倍速い。

```bash
uv run python main.py scrape
```

- 取得済みの race_id は `data/scraped_ids.txt` に記録され、中断後も再開できる
- 馬の過去成績は `data/raw/horses/{horse_id}.parquet` が存在すればスキップされる
- 同時接続数は `config.yaml` の `concurrent_requests` で変更可能（推奨: 5〜8）

馬歴のみ再取得したい場合（レース結果は変えずに馬データだけ取り直すとき）：

```bash
rm data/raw/horses/*.parquet
uv run python main.py scrape_horses
```

既存のレース parquet から馬IDを収集し、馬歴のみ取得する。

### ステップ 2: 特徴量構築

スクレイピングデータから学習用特徴量を生成し `data/processed/features_train.parquet` に保存する。

```bash
uv run python main.py features
```

### ステップ 3: 評価モデルの学習

train/test 分割ありで LightGBM を学習する。精度評価とハイパーパラメータ調整に使用する。

```bash
uv run python main.py train
```

- `train_years` で学習、`test_year` でテスト（デフォルト: 2020-2024 学習 / 2025 テスト）
- early stopping で最適イテレーション数を自動決定
- `data/processed/models/lgb_model.txt` に保存
- 学習後に特徴量重要度 Top20 を表示

### ステップ 4: 本番モデルの再学習

評価モデルで精度を確認した後、全データを使って本番用モデルを再学習する。

```bash
uv run python main.py retrain
```

- 評価モデルの `best_iteration` をそのまま使用（early stopping 不要）
- 2020年〜取得済み全データで学習
- `data/processed/models/lgb_model_production.txt` に保存
- `predict` コマンドは本番モデルが存在すれば自動的に優先して使用する

> **注意**: `retrain` は `train` を実行済みでないと動作しない。

### ステップ 5: レース一覧の確認

予測したいレースの race_id を調べる。

```bash
# 今日のレース一覧
uv run python main.py list_races

# 特定日を指定（YYYY-MM-DD 形式）
uv run python main.py list_races 2026-04-12
```

出力例：
```
2026-04-12 のレース一覧:

race_id           場所    回   日  R
------------------------------------
202603060601      福島   3回   6日  1R
202606030601      中山   3回   6日  1R
202609020601      阪神   2回   6日  1R
...
202609020611      阪神   2回   6日  11R
```

### ステップ 6: レース予測と買い目算出

race_id を指定して予測と買い目を出力する。開催前レース（出馬表のみ存在）にも対応している。

```bash
uv run python main.py predict 202405050111
```

出力例：
```
============================================================
レース: 202405050111
============================================================

【モデル予測上位5頭】
 horse_num horse_name  win_prob top3_prob  raw_score
         3     ○○○○    18.3%     52.1%      2.41
         7     △△△△    15.7%     48.3%      2.18
        ...

【買い目】（合計: 3,000円）
 bet_type combination  odds    ev  kelly_ratio  bet_amount
     単勝           3   5.2  0.952        1.83%      1,800円
     複勝           7   2.1  1.131        3.21%      1,200円
```

### ステップ 7: 回収率サマリー

過去の買い目履歴と回収率を集計する。

```bash
uv run python main.py summary
```

---

## 推奨ワークフロー

```
scrape → features → train → (評価・調整) → retrain → list_races → predict
                               ↑
                        精度が不十分なら
                        特徴量・ハイパーパラメータを調整して再実行
```

新しいデータが溜まったら `scrape` → `features` → `retrain` だけで本番モデルを更新できる。

---

## 設定（config.yaml）

```yaml
scraper:
  target_years: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
  target_until: "2026-03"      # この年月まで取得
  request_interval_min: 1.0   # リクエスト間隔（秒）最小
  request_interval_max: 2.5   # リクエスト間隔（秒）最大
  concurrent_requests: 5      # 同時リクエスト数（増やすほど速いがBANリスク上昇）

model:
  train_years: [2020, 2021, 2022, 2023, 2024]  # 評価モデルの学習データ
  test_year: 2025                               # 評価モデルのテストデータ

betting:
  kelly_fraction: 0.25        # Fractional Kelly の係数
  min_kelly_threshold: 0.02   # Kelly 比率がこれ未満は見送り
  min_expected_value: 1.1     # 期待値 110% 未満は見送り
  max_bet_ratio_per_race: 0.20  # 1レース最大賭け比率（総資金の20%）
  bankroll: 100000            # 初期資金（円）
```

---

## モデルの詳細

### 目的変数

LightGBM の `lambdarank` を使用し、着順をランキング問題として解く。
ラベルは `rank_score = 19 - 着順`（1着=18点、18着=1点）に変換して学習する。

### 特徴量（27種）

| カテゴリ | 特徴量 |
|---|---|
| コース条件 | 芝/ダート、回り、距離、馬場状態、天候、競馬場、グレード |
| 出走情報 | 頭数、枠番、馬番、斤量 |
| 馬 | 馬齢、通算勝率、通算複勝率、直近3走複勝率、前走着順、休養日数、上がり3F平均、出走回数、脚質スコア、馬体重・増減 |
| 騎手 | 直近30日勝率、直近30日複勝率、コース別勝率、直近騎乗数 |

### 時系列分割

未来情報リークを防ぐため、年度単位の時系列分割を採用している。
特徴量生成時もレース開催日より前のデータのみを参照する。

| モデル | 学習データ | テストデータ | 用途 |
|---|---|---|---|
| 評価モデル | 2020〜2024年 | 2025年 | 精度評価・ハイパーパラメータ調整 |
| 本番モデル | 2020〜2026年3月（全データ） | なし | 実際の予測に使用 |

### 評価指標

- **NDCG@3**: 上位3頭の予測精度（メイン指標）
- **AUC（単勝）**: 1着馬の識別精度
- **AUC（複勝）**: 3着以内馬の識別精度
- **回収率シミュレーション**: モデル1位予測馬を毎レース単勝購入した場合の回収率

---

## 買い目最適化の詳細

### Kelly Criterion

モデルの推定勝率 $p$ とオッズ $b$（払戻倍率）から最適賭け比率 $f^*$ を算出する。

$$f^* = \frac{b \cdot p - (1 - p)}{b}$$

実装では **Quarter Kelly**（$f^* \times 0.25$）を使用する。フルKellyは理論上最適だが分散が大きく、連敗時の資金減少が激しいため、25%に縮小することでリスクを抑える。

### 期待値フィルター

期待値 $E = p \times b$ が 1.1（110%）未満の買い目は除外する。
控除率（約25%）を考慮すると、期待値 1.0 付近の買い目は長期的にマイナスになるためエッジが必要。

### 対応馬券種

| 馬券 | 確率の推定方法 |
|---|---|
| 単勝 | モデルの勝率 $p_i$ をそのまま使用 |
| 複勝 | Plackett-Luce モデルの Monte Carlo シミュレーション（10,000回）で推定 |
| 馬連 | $P(A \cap B \text{ が1・2着}) \approx p_A \cdot p_B / (1-p_A) + p_B \cdot p_A / (1-p_B)$ |
| 3連複 | 3頭の複勝確率の積の平方根（相関を考慮した近似） |

### 資金管理

- 1レースの合計賭け比率に上限（デフォルト: 総資金の20%）を設定
- 上限超過時は全買い目のKelly比率を比例縮小する

---

## Notebooks

| ファイル | 内容 |
|---|---|
| `01_data_exploration.ipynb` | スクレイピングデータの基本統計・欠損確認 |
| `02_feature_analysis.ipynb` | 特徴量と勝率の相関・特徴量間の相関行列 |
| `03_model_evaluation.ipynb` | 評価指標・特徴量重要度・市場予測との比較 |

```bash
uv run jupyter notebook notebooks/
```
