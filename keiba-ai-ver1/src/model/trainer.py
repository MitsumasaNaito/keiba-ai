"""
LightGBM モデルの学習
時系列クロスバリデーションを用いてモデルを訓練する
"""
import logging

import lightgbm as lgb
import pandas as pd

from .model_store import ModelStore
from ..features.feature_pipeline import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: dict):
        self.cfg = config["model"]
        self.store = ModelStore(config)

    def train(self, df: pd.DataFrame) -> lgb.Booster:
        """
        時系列分割で学習
        train_years で学習し test_year で評価する
        """
        train_years = self.cfg["train_years"]
        test_year = self.cfg["test_year"]

        df = df.copy()
        df["race_date"] = pd.to_datetime(df.get("race_date"), errors="coerce")
        # race_date が空の場合は race_id 先頭4桁から年を補完
        id_year = df["race_id"].str[:4].astype(int, errors="ignore")
        df["year"] = df["race_date"].dt.year.fillna(id_year).astype(int)

        train_df = df[df["year"].isin(train_years)].copy()
        test_df = df[df["year"] == test_year].copy()

        logger.info(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        X_train = train_df[FEATURE_COLUMNS].astype(float)
        y_train = train_df["rank_score"].astype(float)
        group_train = train_df.groupby("race_id").size().values

        X_test = test_df[FEATURE_COLUMNS].astype(float)
        y_test = test_df["rank_score"].astype(float)
        group_test = test_df.groupby("race_id").size().values

        dtrain = lgb.Dataset(
            X_train,
            label=y_train,
            group=group_train,
            feature_name=FEATURE_COLUMNS,
        )
        dtest = lgb.Dataset(
            X_test,
            label=y_test,
            group=group_test,
            feature_name=FEATURE_COLUMNS,
            reference=dtrain,
        )

        params = {
            "objective": self.cfg["objective"],
            "metric": self.cfg["metric"],
            "ndcg_eval_at": self.cfg["ndcg_eval_at"],
            "learning_rate": self.cfg["learning_rate"],
            "num_leaves": self.cfg["num_leaves"],
            "min_child_samples": self.cfg["min_child_samples"],
            "feature_fraction": self.cfg["feature_fraction"],
            "bagging_fraction": self.cfg["bagging_fraction"],
            "bagging_freq": self.cfg["bagging_freq"],
            "lambda_l1": self.cfg["lambda_l1"],
            "lambda_l2": self.cfg["lambda_l2"],
            "verbose": self.cfg["verbose"],
        }

        callbacks = [
            lgb.early_stopping(self.cfg["early_stopping_rounds"], verbose=True),
            lgb.log_evaluation(100),
        ]

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=self.cfg["n_estimators"],
            valid_sets=[dtest],
            callbacks=callbacks,
        )

        self.store.save(model)
        logger.info(f"評価モデル保存完了 (best_iteration={model.best_iteration})")
        return model

    def retrain(self, df: pd.DataFrame) -> lgb.Booster:
        """
        全データで再学習して本番モデルを生成する
        イテレーション数は評価モデルの best_iteration を使用する
        """
        if not self.store.exists(production=False):
            raise FileNotFoundError("先に train コマンドで評価モデルを作成してください")

        eval_model = self.store.load(production=False)
        best_iteration = eval_model.best_iteration
        if best_iteration <= 0:
            best_iteration = eval_model.num_trees()
        logger.info(f"評価モデルの best_iteration: {best_iteration}")

        df = df.copy()
        df["race_date"] = pd.to_datetime(df["race_date"])

        X_all = df[FEATURE_COLUMNS].astype(float)
        y_all = df["rank_score"].astype(float)
        group_all = df.groupby("race_id").size().values

        logger.info(f"全データで再学習: {len(df)} エントリ, {df['race_id'].nunique()} レース")

        dall = lgb.Dataset(
            X_all,
            label=y_all,
            group=group_all,
            feature_name=FEATURE_COLUMNS,
        )

        params = {
            "objective": self.cfg["objective"],
            "metric": self.cfg["metric"],
            "ndcg_eval_at": self.cfg["ndcg_eval_at"],
            "learning_rate": self.cfg["learning_rate"],
            "num_leaves": self.cfg["num_leaves"],
            "min_child_samples": self.cfg["min_child_samples"],
            "feature_fraction": self.cfg["feature_fraction"],
            "bagging_fraction": self.cfg["bagging_fraction"],
            "bagging_freq": self.cfg["bagging_freq"],
            "lambda_l1": self.cfg["lambda_l1"],
            "lambda_l2": self.cfg["lambda_l2"],
            "verbose": self.cfg["verbose"],
        }

        model = lgb.train(
            params,
            dall,
            num_boost_round=best_iteration,
            callbacks=[lgb.log_evaluation(100)],
        )

        self.store.save_production(model)
        logger.info("本番モデル保存完了")
        return model

    def get_feature_importance(self, model: lgb.Booster) -> pd.DataFrame:
        importance = model.feature_importance(importance_type="gain")
        return pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": importance,
        }).sort_values("importance", ascending=False)
