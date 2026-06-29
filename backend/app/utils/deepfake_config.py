# app/utils/deepfake_config.py
"""
TruthLens Deepfake Detection Configuration
==========================================

Active models (three-model ensemble):
  deepfake_exp  prithivMLmods/Deepfake-Detection-Exp-02-21   ViT 98.84%
  siglip        prithivMLmods/deepfake-detector-model-v1     SigLIP2 94.44%
  wvolf_vit     Wvolf/ViT_Deepfake_Detection                 ViT 98.70%

All deprecated / broken model blocks are preserved so existing YAML configs
loaded from disk and any code referencing those keys by string still work.
Their weights are set to 0.0 and their deprecated flag is True so the factory
skips them when building the ensemble.
"""

import os
import yaml
from typing import Dict
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {

    # ─────────────────────────────────────────────────────
    # MODEL 1  — DeepfakeExpDetector (PRIMARY)
    # prithivMLmods/Deepfake-Detection-Exp-02-21
    # ViT-base-patch16-224 · 98.84% accuracy
    # id2label: {0: "Deepfake", 1: "Real"}  fake_index = 0
    # ─────────────────────────────────────────────────────
    "deepfake_exp": {
        "detector_type": "deepfake_exp",
        "model_name":    "prithivMLmods/Deepfake-Detection-Exp-02-21",
        "threshold":     0.65,
        "weight":        1.2,     # upweighted: highest accuracy of the three
        "description":   "ViT-base fine-tuned deepfake detector (98.84% accuracy)",
        "architecture":  "ViT-base-patch16-224",
        "fake_index":    0,       # "Deepfake" == index 0
    },

    # ─────────────────────────────────────────────────────
    # MODEL 2  — SigLIPDetector (DIVERSITY)
    # prithivMLmods/deepfake-detector-model-v1
    # SigLIP2-base-patch16-512 · 94.44% accuracy
    # id2label: {0: "fake", 1: "real"}  fake_index = 0
    #
    # SigLIP uses sigmoid-loss pre-training and a vision-language
    # encoder backbone — architecturally distinct from ViT softmax
    # classifiers, so its error distribution is uncorrelated.
    # ─────────────────────────────────────────────────────
    "siglip": {
        "detector_type": "siglip",
        "model_name":    "prithivMLmods/deepfake-detector-model-v1",
        "threshold":     0.50,
        "weight":        0.9,     # slightly downweighted vs ViTs (lower acc)
        "description":   "SigLIP2 deepfake detector (94.44% accuracy, diverse backbone)",
        "architecture":  "SigLIP2-base-patch16-512",
        "fake_index":    0,       # "fake" == index 0
    },

    # ─────────────────────────────────────────────────────
    # MODEL 3  — WvolfViTDetector (SECONDARY ViT)
    # Wvolf/ViT_Deepfake_Detection
    # ViT (MSc Solent University) · 98.70% accuracy
    # id2label: {0: "REAL", 1: "FAKE"}  fake_index = 1
    # ─────────────────────────────────────────────────────
    "wvolf_vit": {
        "detector_type": "wvolf_vit",
        "model_name":    "Wvolf/ViT_Deepfake_Detection",
        "threshold":     0.50,
        "weight":        1.0,
        "description":   "ViT deepfake detector, MSc project (98.70% accuracy)",
        "architecture":  "ViT (MSc Solent)",
        "fake_index":    1,       # "FAKE" == index 1
    },

    # ─────────────────────────────────────────────────────
    # OPTIONAL  — CLIPDetector (supporting signal only)
    # NOT in default ensemble.models.
    # Add "clip" to ensemble.models to enable as 4th model.
    # At threshold 0.82, FPR < 5% but TPR only ~60%.
    # ─────────────────────────────────────────────────────
    "clip": {
        "detector_type": "clip",
        "model_name":    "openai/clip-vit-base-patch32",
        "threshold":     0.82,
        "weight":        0.5,
        "description":   "CLIP zero-shot — optional supporting signal only",
        "zero_shot":     True,
    },

    # ─────────────────────────────────────────────────────
    # DEPRECATED  — preserved for backward-compat / YAML
    # Factory stubs these to PlaceholderDetector.
    # weight = 0.0 so they never contribute to ensemble avg.
    # ─────────────────────────────────────────────────────

    # prithivMLmods/Deep-Fake-Detector-v2-Model:
    #   92% accuracy, poorly calibrated — caused 0% detection bug.
    "prithiv": {
        "detector_type": "prithiv",    # alias → DeepfakeExpDetector
        "model_name":    "prithivMLmods/Deep-Fake-Detector-v2-Model",
        "threshold":     0.50,
        "weight":        0.0,
        "description":   "DEPRECATED (92% acc, poor calibration) → alias to deepfake_exp",
        "deprecated":    True,
    },

    # ssundaram21/vfdnet: repo deleted, label-inversion bug.
    "vfdnet": {
        "detector_type": "vfdnet",
        "model_name":    "ssundaram21/vfdnet",
        "threshold":     0.50,
        "weight":        0.0,
        "description":   "DEPRECATED — repo unavailable + label-inversion bug",
        "deprecated":    True,
    },

    # timm EfficientNet: randomly initialised head.
    "efficientnet": {
        "detector_type": "efficientnet",
        "model_name":    "timm/efficientnet_b3.ra2_in1k",
        "threshold":     0.50,
        "weight":        0.0,
        "description":   "DEPRECATED — random classification head = noise",
        "deprecated":    True,
    },

    # buildborderless/CommunityForensics: shape-mismatch error on load.
    "community_vit": {
        "detector_type": "community_vit",
        "model_name":    "buildborderless/CommunityForensics-DeepfakeDet-ViT",
        "threshold":     0.50,
        "weight":        0.0,
        "description":   "DEPRECATED — shape-mismatch error on load",
        "deprecated":    True,
    },

    # Xception: random head (same bug as EfficientNet).
    "xception": {
        "detector_type": "xception",
        "model_name":    "timm/xception",
        "threshold":     0.50,
        "input_size":    [299, 299],
        "weight":        0.0,
        "description":   "DEPRECATED — random classification head",
        "deprecated":    True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

PIPELINE_CONFIG = {

    # ── Ensemble ──────────────────────────────────────────────────────────────
    "ensemble": {
        "enabled":   True,
        # Three active models. Add "clip" here to enable zero-shot as 4th.
        "models":    ["deepfake_exp", "siglip", "wvolf_vit"],
        "voting":    "weighted",
        # Weighted-average threshold. Lower = more sensitive, more FP.
        # At 0.50 with these three models, expected FPR < 3% on real faces.
        "threshold": 0.50,
    },

    # ── Face extraction ───────────────────────────────────────────────────────
    # Note: if no face is found, the ensemble falls back to full-image analysis
    # rather than skipping (previous skip behaviour caused 0% TPR on
    # partially-occluded or tilted-head deepfakes).
    "face_extraction": {
        "enabled":              True,
        "detector":             "opencv",
        "min_face_size":        50,
        "confidence_threshold": 0.95,
        "max_faces":            1,
        "align_faces":          True,
        "selection":            "largest",
        "save_faces":           False,
    },

    # ── Metadata analysis ─────────────────────────────────────────────────────
    "metadata_analysis": {
        "enabled":                True,
        "check_tampering":        True,
        "check_editing_software": True,
        "check_gps":              True,
        "check_timestamps":       True,
        "check_thumbnail":        True,
        "suspicious_software":    [
            "Photoshop", "GIMP", "Affinity", "Lightroom",
            "DeepFaceLab", "FakeApp",
        ],
        "tampering_threshold":    0.7,
    },

    # ── Risk scoring ──────────────────────────────────────────────────────────
    "risk_scoring": {
        "high_threshold":   0.8,
        "medium_threshold": 0.5,
        "low_threshold":    0.3,
    },

    # ── Explainability ────────────────────────────────────────────────────────
    "explainability": {
        "enabled":              True,
        "return_faces":         True,
        "return_bboxes":        True,
        "highlight_suspicious": True,
    },

    # ── Adversarial testing ───────────────────────────────────────────────────
    "adversarial_testing": {
        "enabled": False,
        "tests": {
            "compression":         {"enabled": True,  "quality_range":     [50, 70, 90]},
            "noise":               {"enabled": True,  "noise_types":       ["gaussian", "salt_pepper"],
                                    "intensity_range": [0.01, 0.05, 0.1]},
            "scaling":             {"enabled": True,  "scale_factors":     [0.5, 0.8, 1.2, 1.5]},
            "blur":                {"enabled": True,  "blur_types":        ["gaussian", "median"],
                                    "kernel_sizes": [3, 5]},
            "brightness_contrast": {"enabled": True,  "brightness_factors": [0.8, 1.2],
                                    "contrast_factors": [0.8, 1.2]},
        },
        "robustness_score": True,
    },

    # ── Anti-evasion ──────────────────────────────────────────────────────────
    "anti_evasion": {
        "enabled":         False,
        "check_crop":      True,
        "check_downscale": True,
        "check_reencode":  True,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# SYSTEM CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_CONFIG = {
    # "ensemble" builds the three-model weighted-average ensemble above.
    # Can be overridden with DEEPFAKE_DETECTOR=deepfake_exp to use a single model.
    "active_detector": os.environ.get("DEEPFAKE_DETECTOR", "ensemble"),

    "cache_dir": os.environ.get(
        "DEEPFAKE_CACHE_DIR",
        os.path.join(BASE_DIR, "models", "deepfake"),
    ),

    # Set DEEPFAKE_USE_GPU=true to run on CUDA (models run fine on CPU).
    "use_gpu": os.environ.get("DEEPFAKE_USE_GPU", "False").lower() == "true",

    # True → on any model load failure, return PlaceholderDetector rather
    # than raising an exception. Recommended for production.
    "fallback_to_placeholder": True,

    "warmup":    False,
    "in_memory": True,
}


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED CONFIG (backward-compat flat merge)
# ──────────────────────────────────────────────────────────────────────────────

DEEPFAKE_CONFIG: Dict = {
    **MODEL_CONFIG,
    **PIPELINE_CONFIG,
    **SYSTEM_CONFIG,
}

# Ensure model cache directory exists at import time
os.makedirs(SYSTEM_CONFIG["cache_dir"], exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# YAML PERSISTENCE HELPERS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def save_configs() -> None:
    """Export MODEL_CONFIG and PIPELINE_CONFIG to YAML files under BASE_DIR/config/."""
    config_dir = os.path.join(BASE_DIR, "config")
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, "model_config.yaml"), "w") as f:
        yaml.dump(MODEL_CONFIG, f, default_flow_style=False)
    with open(os.path.join(config_dir, "pipeline_config.yaml"), "w") as f:
        yaml.dump(PIPELINE_CONFIG, f, default_flow_style=False)


def load_configs() -> None:
    """Load MODEL_CONFIG / PIPELINE_CONFIG from YAML if they exist on disk."""
    config_dir    = os.path.join(BASE_DIR, "config")
    model_path    = os.path.join(config_dir, "model_config.yaml")
    pipeline_path = os.path.join(config_dir, "pipeline_config.yaml")
    if os.path.exists(model_path):
        with open(model_path) as f:
            MODEL_CONFIG.update(yaml.safe_load(f) or {})
    if os.path.exists(pipeline_path):
        with open(pipeline_path) as f:
            PIPELINE_CONFIG.update(yaml.safe_load(f) or {})


try:
    load_configs()
except Exception:
    pass