"""
モデルの保存・ロード管理
"""
import logging
from pathlib import Path

import lightgbm as lgb

logger = logging.getLogger(__name__)

MODEL_FILENAME = "lgb_model.txt"
MODEL_FILENAME_PRODUCTION = "lgb_model_production.txt"


class ModelStore:
    def __init__(self, config: dict):
        self.model_dir = Path(config["paths"]["models"])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / MODEL_FILENAME
        self.production_path = self.model_dir / MODEL_FILENAME_PRODUCTION

    def save(self, model: lgb.Booster):
        model.save_model(str(self.model_path))
        logger.info(f"評価モデル保存: {self.model_path}")

    def save_production(self, model: lgb.Booster):
        model.save_model(str(self.production_path))
        logger.info(f"本番モデル保存: {self.production_path}")

    def load(self, production: bool = False) -> lgb.Booster:
        """
        production=True: 本番モデルを読み込む（存在しない場合は評価モデルにフォールバック）
        production=False: 評価モデルを読み込む
        """
        if production and self.production_path.exists():
            path = self.production_path
        elif self.model_path.exists():
            path = self.model_path
        else:
            raise FileNotFoundError(f"モデルが見つかりません: {self.model_dir}")
        model = lgb.Booster(model_file=str(path))
        logger.info(f"モデル読み込み: {path}")
        return model

    def exists(self, production: bool = False) -> bool:
        path = self.production_path if production else self.model_path
        return path.exists()
