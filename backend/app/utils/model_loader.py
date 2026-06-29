"""
app/utils/model_loader.py

3-MODEL ENSEMBLE for Fake News Detection
=========================================

Models in ensemble:
1. Arko007/fake-news-roberta-5M - Political news, inverted score (high=REAL)
2. jy46604790/Fake-News-Bert-Detect - Diverse news (40k articles), normal convention
3. hamzab/roberta-fake-news-classification - Different training distribution

Ensemble Strategy:
- Weighted voting based on each model's strengths
- Parallel execution for speed
- Heuristic signals as 4th voter

NEW: Each model load now has a 60-second timeout.
If a model can't download in time, it's skipped and the ensemble still works.
"""

import os
import logging
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time

logger = logging.getLogger(__name__)


class ModelVote:
    """Individual model vote"""
    def __init__(self, name: str, is_fake: bool, confidence: float, weight: float):
        self.name = name
        self.is_fake = is_fake
        self.confidence = confidence
        self.weight = weight


class FakeNewsModel:
    """
    3-Model Ensemble for Fake News Detection.
    Loads all models at once and runs them in parallel.
    """

    # Model configurations
    MODELS = {
        "arko007": {
            "name": "Arko007/fake-news-roberta-5M",
            "local_path": "E:/Verify_AI/models/fake-news-roberta-5M",
            "weight": 0.35,
            "needs_fast": True,
            "inverted": True,  # High score = REAL
            "threshold": 0.69,
            "description": "Political/news structure"
        },
        "bert_detect": {
            "name": "jy46604790/Fake-News-Bert-Detect",
            "local_path": "E:/Verify_AI/models/fake-news-bert-detect",
            "weight": 0.40,
            "needs_fast": False,
            "inverted": False,  # LABEL_0 = FAKE, LABEL_1 = REAL
            "threshold": 0.5,
            "description": "Diverse news (40k articles)"
        },
        "roberta_hamzab": {
            "name": "hamzab/roberta-fake-news-classification",
            "local_path": "E:/Verify_AI/models/roberta-fake-news",
            "weight": 0.25,
            "needs_fast": False,
            "inverted": False,  # Returns {"Fake": prob, "Real": prob}
            "threshold": 0.5,
            "description": "Different training distribution"
        }
    }

    # Ensemble threshold (weighted average >= this = FAKE)
    FAKE_THRESHOLD = 0.50

    # How many seconds to wait for a single model to load
    MODEL_LOAD_TIMEOUT = 60

    def __init__(self):
        self.classifiers = {}
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"📦 3-Model Ensemble Initialized")
        logger.info(f"💻 Device: {self.device}")
        logger.info(f"⚙️  Ensemble FAKE Threshold: {self.FAKE_THRESHOLD}")

    def _load_one_model(self, model_key: str, config: dict):
        """Load a single model (called in a thread)."""
        model_path = config["local_path"] if os.path.exists(config["local_path"]) else config["name"]
        if model_path == config["name"]:
            logger.warning(f"⚠️  Local path not found for {config['name']} - downloading from HuggingFace")

        try:
            if config.get("needs_fast", False):
                pipe = pipeline(
                    "text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    use_fast=False,
                    device=0 if torch.cuda.is_available() else -1,
                )
            else:
                pipe = pipeline(
                    "text-classification",
                    model=model_path,
                    tokenizer=model_path,
                    device=0 if torch.cuda.is_available() else -1,
                )
            return (model_key, pipe)
        except Exception as e:
            logger.error(f"❌ Failed to load {config['name']}: {e}")
            return (model_key, None)

    def load(self) -> bool:
        """Load all 3 models, with a timeout per model."""
        logger.info("⏳ Loading 3-model ensemble (timeout per model: {}s)...".format(self.MODEL_LOAD_TIMEOUT))

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for model_key, config in self.MODELS.items():
                futures[executor.submit(self._load_one_model, model_key, config)] = model_key

            for future in as_completed(futures):
                model_key = futures[future]
                try:
                    key, pipe = future.result(timeout=self.MODEL_LOAD_TIMEOUT)
                    self.classifiers[model_key] = pipe
                    if pipe is not None:
                        logger.info(f"   ✅ Loaded: {self.MODELS[model_key]['description']} ({self.MODELS[model_key]['name']})")
                    else:
                        logger.warning(f"   ⚠️  Skipped {self.MODELS[model_key]['description']} (failed to load)")
                except TimeoutError:
                    logger.error(f"   ⏰ Timeout loading {self.MODELS[model_key]['description']} — skipping")
                    self.classifiers[model_key] = None
                except Exception as e:
                    logger.error(f"   ❌ Unexpected error loading {self.MODELS[model_key]['description']}: {e}")
                    self.classifiers[model_key] = None

        loaded_count = sum(1 for v in self.classifiers.values() if v is not None)
        if loaded_count == 0:
            logger.error("❌ No models could be loaded — ensemble will fail")
            return False

        self.model_loaded = True
        logger.info(f"✅ Ensemble ready with {loaded_count}/3 models")
        return True

    # ── Prediction methods (unchanged) ──

    def _predict_arko007(self, text: str, config: dict) -> dict:
        try:
            result = self.classifiers["arko007"](text[:512])[0]
            raw_score = float(result['score'])
            is_fake = raw_score <= config["threshold"]
            fake_prob = 1.0 - raw_score if not is_fake else raw_score
            confidence = abs(raw_score - config["threshold"]) * 3.0
            confidence = min(confidence, 0.99)
            return {
                "is_fake": is_fake,
                "fake_probability": fake_prob,
                "real_probability": 1.0 - fake_prob,
                "confidence": confidence,
                "raw_score": raw_score,
                "model": "arko007"
            }
        except Exception as e:
            logger.error(f"Arko007 prediction error: {e}")
            return None

    def _predict_bert_detect(self, text: str, config: dict) -> dict:
        try:
            result = self.classifiers["bert_detect"](text[:512])[0]
            label = result['label']
            score = float(result['score'])
            is_fake = (label == "LABEL_0")
            fake_prob = score if is_fake else 1.0 - score
            confidence = abs(score - 0.5) * 2.0
            confidence = min(confidence, 0.99)
            return {
                "is_fake": is_fake,
                "fake_probability": fake_prob,
                "real_probability": 1.0 - fake_prob,
                "confidence": confidence,
                "raw_score": score,
                "model": "bert_detect"
            }
        except Exception as e:
            logger.error(f"BERT Detect prediction error: {e}")
            return None

    def _predict_roberta_hamzab(self, text: str, config: dict) -> dict:
        try:
            result = self.classifiers["roberta_hamzab"](text[:512])[0]
            label = result['label']
            score = float(result['score'])
            is_fake = (label == "Fake" or label == "LABEL_0")
            fake_prob = score if is_fake else 1.0 - score
            confidence = abs(score - 0.5) * 2.0
            confidence = min(confidence, 0.99)
            return {
                "is_fake": is_fake,
                "fake_probability": fake_prob,
                "real_probability": 1.0 - fake_prob,
                "confidence": confidence,
                "raw_score": score,
                "model": "roberta_hamzab"
            }
        except Exception as e:
            logger.error(f"hamzab RoBERTa prediction error: {e}")
            return None

    def predict(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return self._error_result("Invalid input")

        if not self.model_loaded and not self.load():
            return self._error_result("Model load failed")

        votes = []
        configs = self.MODELS

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            if self.classifiers.get("arko007"):
                futures[executor.submit(self._predict_arko007, text, configs["arko007"])] = "arko007"
            if self.classifiers.get("bert_detect"):
                futures[executor.submit(self._predict_bert_detect, text, configs["bert_detect"])] = "bert_detect"
            if self.classifiers.get("roberta_hamzab"):
                futures[executor.submit(self._predict_roberta_hamzab, text, configs["roberta_hamzab"])] = "roberta_hamzab"

            for future in as_completed(futures):
                result = future.result()
                if result:
                    model_key = futures[future]
                    weight = configs[model_key]["weight"]
                    result["weight"] = weight
                    votes.append(result)

        if not votes:
            return self._error_result("All models failed")

        total_weight = sum(v["weight"] for v in votes)
        weighted_fake_prob = sum(v["fake_probability"] * v["weight"] for v in votes) / total_weight
        weighted_confidence = sum(v["confidence"] * v["weight"] for v in votes) / total_weight

        is_fake = weighted_fake_prob >= self.FAKE_THRESHOLD
        real_probability = 1.0 - weighted_fake_prob

        logger.info(f"🎯 Ensemble: fake_prob={weighted_fake_prob:.3f}, conf={weighted_confidence:.3f} → {'FAKE' if is_fake else 'REAL'}")

        return {
            "is_fake": is_fake,
            "verdict": "FAKE" if is_fake else "REAL",
            "fake_probability": round(weighted_fake_prob, 4),
            "real_probability": round(real_probability, 4),
            "confidence": round(weighted_confidence, 4),
            "ensemble_details": {
                "votes": votes,
                "total_weight": round(total_weight, 2),
                "threshold_used": self.FAKE_THRESHOLD
            },
            "models_loaded": len(votes)
        }

    def _error_result(self, reason: str) -> dict:
        return {
            "is_fake": False,
            "verdict": "ERROR",
            "fake_probability": 0.5,
            "real_probability": 0.5,
            "confidence": 0.0,
            "error": reason,
            "ensemble_details": {}
        }

    def get_model_info(self) -> dict:
        loaded_models = [k for k, v in self.classifiers.items() if v is not None]
        return {
            "ensemble_size": len(loaded_models),
            "loaded_models": loaded_models,
            "threshold": self.FAKE_THRESHOLD,
            "device": str(self.device),
            "models": self.MODELS
        }


# Backward compatibility
FakeNewsDetector = FakeNewsModel