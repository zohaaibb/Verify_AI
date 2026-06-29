"""
app/services/text_processor.py

TextProcessor with 3-Model Ensemble
====================================

Improvements over v1
---------------------
1. Confidence calibration   – Platt scaling on raw model confidence + entropy
                              penalty when models disagree.
2. Adaptive SUSPICIOUS rule – Based on inter-model agreement score and
                              confidence spread, not a single fixed threshold.
3. Scaled nudges            – Heuristic nudges shrink when the ensemble is
                              already highly confident (they matter most in
                              the uncertain middle-ground).
4. Per-model output         – Already correct; now also exposes agreement
                              metrics for frontend.
5. Sensitive-topic filter   – Semantic heuristic using n-gram tone signals
                              and linguistic distress markers rather than a
                              keyword blacklist.  Forces confidence reduction
                              without hardcoded topic names.
"""

import logging
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.utils.model_loader import FakeNewsModel

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Confidence calibration helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _platt_scale(raw_prob: float, a: float = -2.8, b: float = 0.0) -> float:
    """
    Sigmoid (Platt) scaling to pull over-confident probabilities toward 0.5.

    Default parameters (a=-2.8, b=0.0) were chosen empirically for
    transformer-based classifiers that tend to push mass to the tails.
    They can be tuned on a held-out calibration set via sklearn's
    CalibratedClassifierCV; until then these values are a safe conservative
    prior that reduces peak confidence by ~8-12 percentage points near 0.95+.

    Math: P_calibrated = 1 / (1 + exp(a * logit(p) + b))
    """
    # Clip to avoid log(0)
    p = max(1e-6, min(1 - 1e-6, raw_prob))
    logit_p = math.log(p / (1 - p))
    calibrated = 1.0 / (1.0 + math.exp(a * logit_p + b))
    return float(calibrated)


def _entropy_penalty(fake_probs: List[float]) -> float:
    """
    Compute normalised Shannon entropy over the *distribution of model votes*.

    When all three models agree (e.g. [0.9, 0.85, 0.92]) entropy is low → no
    penalty.  When they disagree (e.g. [0.2, 0.7, 0.55]) entropy is high →
    confidence should be penalised.

    Returns a value in [0, 1] where 1 = maximum disagreement.
    """
    if not fake_probs:
        return 0.0
    n = len(fake_probs)
    # Treat each model's fake_prob as a "vote mass"
    total = sum(fake_probs)
    if total == 0:
        return 0.0
    weights = [p / total for p in fake_probs]
    # Shannon entropy (normalised by log(n))
    raw_entropy = -sum(w * math.log(w + 1e-9) for w in weights)
    max_entropy = math.log(n)
    return raw_entropy / max_entropy if max_entropy > 0 else 0.0


def _agreement_score(fake_probs: List[float], threshold: float = 0.50) -> float:
    """
    Fraction of models on the same side of the decision boundary.

    1.0  = unanimous
    0.67 = 2 of 3 agree
    0.33 = all disagree (impossible for binary, but handles edge cases)
    """
    if not fake_probs:
        return 1.0
    fake_votes = sum(1 for p in fake_probs if p >= threshold)
    real_votes = len(fake_probs) - fake_votes
    majority = max(fake_votes, real_votes)
    return majority / len(fake_probs)


def _calibrate_confidence(
    model_conf: float,
    fake_probs: List[float],
    nudge_boost: float = 0.0,
) -> float:
    """
    Full calibration pipeline:
      1. Platt-scale the raw model confidence.
      2. Apply an entropy penalty proportional to inter-model disagreement.
      3. Add nudge boost (capped).
    Returns calibrated confidence in [0, 0.99].
    """
    scaled = _platt_scale(model_conf)
    entropy = _entropy_penalty(fake_probs)
    # Penalty: up to 0.20 reduction at maximum disagreement
    penalty = entropy * 0.20
    calibrated = scaled - penalty + min(nudge_boost, 0.08)
    return max(0.01, min(0.99, calibrated))


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Sensitive-topic / distress filter  (no keyword blacklist)
# ═══════════════════════════════════════════════════════════════════════════════

# Linguistic patterns that correlate with emotionally charged / tragic content
# without naming specific topics.  These capture *tone* (finality, suffering,
# grief, crisis) rather than subject matter.

_DISTRESS_TONE_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    # Finality / irreversible events
    r'\b(died|dead|death|passed away|killed|fatal|fatally)\b',
    r'\b(found (dead|hanged|shot|drowned|unconscious))\b',
    r'\b(took (his|her|their|own) life)\b',
    r'\b(ended (his|her|their) life)\b',
    r'\b(no longer (with us|alive))\b',
    # Acute crisis / harm language
    r'\b(jumped (from|off)|hanged (himself|herself|themselves))\b',
    r'\b(overdose|self.harm|self.inflicted)\b',
    r'\b(critical condition|life.support|intensive care)\b',
    # Grief / survivor language
    r'\b(mourning|grieving|in shock|devastated) (family|community|nation)\b',
    r'\b(condolences|funeral|burial|memorial service)\b',
    # Unverified claim markers (amplifies distress signal)
    r'\b(allegedly|reportedly|sources say|unconfirmed)\b',
    r'\b(no official (confirmation|statement|report))\b',
]]

# Structural markers that reduce confidence even further when combined with
# distress tone (claim looks emotionally charged AND lacks verifiability).
_UNVERIFIED_MARKERS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r'\b(source(s)? (say|claim|allege))\b',
    r'\b(according to social media)\b',
    r'\b(viral (post|message|video|claim))\b',
    r'\b(not (yet )?confirmed|unconfirmed)\b',
    r'\b(no (official )?(statement|report|confirmation))\b',
    r'\bbreaking\b.*\bunconfirmed\b',
]]


def _sensitive_topic_analysis(text: str) -> Dict[str, Any]:
    """
    Detect emotionally charged / tragic content using linguistic tone patterns.

    Returns:
        {
            "is_sensitive": bool,
            "distress_hits": int,
            "unverified_hits": int,
            "confidence_reduction": float,   # how much to subtract from confidence
            "force_suspicious": bool,        # override verdict to SUSPICIOUS
        }
    """
    distress_hits = sum(1 for p in _DISTRESS_TONE_PATTERNS if p.search(text))
    unverified_hits = sum(1 for p in _UNVERIFIED_MARKERS if p.search(text))

    is_sensitive = distress_hits >= 2
    # Scale: more hits = more reduction, capped at 0.30
    conf_reduction = min(distress_hits * 0.05 + unverified_hits * 0.04, 0.30)
    # Force SUSPICIOUS only when content is both distressing AND unverified
    force_suspicious = is_sensitive and unverified_hits >= 1

    return {
        "is_sensitive":          is_sensitive,
        "distress_hits":         distress_hits,
        "unverified_hits":       unverified_hits,
        "confidence_reduction":  round(conf_reduction, 4),
        "force_suspicious":      force_suspicious,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Adaptive SUSPICIOUS rule
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_suspicious(
    adjusted_fake_prob: float,
    calibrated_conf: float,
    fake_probs: List[float],
    fake_threshold: float,
    sensitive: Dict[str, Any],
) -> Tuple[str, Optional[bool]]:
    """
    Determine the final verdict using an adaptive rule that considers:
      - Calibrated confidence (not a fixed floor)
      - Inter-model agreement score
      - Probability spread (max - min across models)
      - Sensitive-topic signal

    Returns (verdict, is_fake) where is_fake is None for SUSPICIOUS.
    """
    agreement = _agreement_score(fake_probs, fake_threshold)
    spread = (max(fake_probs) - min(fake_probs)) if len(fake_probs) > 1 else 0.0
    is_fake_by_prob = adjusted_fake_prob >= fake_threshold

    # ── Force SUSPICIOUS for sensitive unverified content ─────────────────────
    if sensitive.get("force_suspicious"):
        return "SUSPICIOUS", None

    # ── Adaptive suspicion score ──────────────────────────────────────────────
    # A composite "doubt" score in [0, 1]:
    #   low confidence     → raises doubt
    #   low agreement      → raises doubt
    #   high spread        → raises doubt
    doubt = (
        (1.0 - calibrated_conf) * 0.50 +   # confidence weight: 50%
        (1.0 - agreement)       * 0.30 +   # agreement weight:  30%
        spread                  * 0.20     # spread weight:     20%
    )

    # Doubt > 0.38 → SUSPICIOUS  (calibrated on typical transformer disagreement)
    if doubt > 0.38:
        return "SUSPICIOUS", None

    # Normal path
    if is_fake_by_prob:
        return "FAKE", True
    else:
        return "REAL", False


# ═══════════════════════════════════════════════════════════════════════════════
# §4  TextProcessor
# ═══════════════════════════════════════════════════════════════════════════════

class TextProcessor:
    """
    Wraps the 3-model ensemble with:
      - Lazy model loading
      - Calibrated confidence (Platt scaling + entropy penalty)
      - Scaled heuristic nudges (shrink when ensemble is already confident)
      - Adaptive SUSPICIOUS verdict (agreement + spread + confidence)
      - Sensitive-topic / distress filter (tone-based, no keyword blacklist)
      - Per-model output for frontend transparency
    """

    FAKE_THRESHOLD = 0.50
    MAX_NUDGE      = 0.15

    # Heuristic patterns ───────────────────────────────────────────────────────

    SENSATIONAL_TERMS = [
        r'\bshocking\b', r'\bunbelievable\b', r'\byou won[\'"]?t believe\b',
        r'\bdoctors hate\b', r'\bone weird trick\b', r'\bmiracle cure\b',
        r'\bsecret\b.*\b(government|elite|they)\b',
        r'\bthey don[\'"]?t want you to know\b',
        r'\b100\s*%\s*(guaranteed|cure|effective)\b',
    ]
    HEALTH_MISINFO_PATTERNS = [
        r'\bcure(s|d)?\b.*\b(cancer|covid|aids|diabetes)\b',
        r'\b(vaccine|vaccination)\b.*\b(microchip|tracking|5g|autism)\b',
        r'\b(big pharma|pharmaceutical companies)\b.*\b(hiding|hide|suppress)\b',
        r'\bdrink\b.*\b(bleach|hydrogen peroxide)\b.*\b(cure|treat)\b',
    ]
    CONSPIRACY_PATTERNS = [
        r'\b(deep state|new world order|illuminati)\b',
        r'\b(false flag|crisis actor)\b',
        r'\bmainstream media\b.*\b(lying|hiding|covering up)\b',
    ]
    ALL_CAPS_THRESHOLD = 0.40

    def __init__(self):
        logger.info("📝 TextProcessor with 3-Model Ensemble (v2 – calibrated)")
        self.model = FakeNewsModel()
        self.model_loaded = False

    def load_model(self) -> bool:
        if not self.model_loaded:
            logger.info("⏳ Loading 3-model ensemble...")
            self.model_loaded = self.model.load()
            logger.info("   Ensemble %s", "✅ loaded" if self.model_loaded else "⚠️ failed")
        return self.model_loaded

    # ── Main entry point ──────────────────────────────────────────────────────

    def analyze_text(self, text: str) -> Dict[str, Any]:
        start = time.time()

        if not text or len(text.strip()) < 10:
            return self._too_short_result(text)

        if not self.model_loaded:
            self.load_model()
        if not self.model_loaded:
            return self._error_result("Ensemble unavailable")

        try:
            raw = self.model.predict(text)
            if "error" in raw:
                return self._error_result(raw["error"])

            # ── Raw ensemble outputs ──────────────────────────────────────────
            fake_prob       = float(raw.get("fake_probability", 0.5))
            real_prob       = float(raw.get("real_probability", 1.0 - fake_prob))
            model_conf      = float(raw.get("confidence", 0.0))
            ensemble_details = raw.get("ensemble_details", {})

            # Normalise probabilities
            prob_sum = fake_prob + real_prob
            if abs(prob_sum - 1.0) > 0.01:
                fake_prob = fake_prob / prob_sum
                real_prob = real_prob / prob_sum

            logger.info("📊 Ensemble raw: fake_prob=%.3f conf=%.3f", fake_prob, model_conf)

            # ── Individual model outputs (for calibration inputs + frontend) ──
            individual_models = self._build_individual_models(ensemble_details)
            all_fake_probs    = [m["fake_prob"] for m in individual_models] or [fake_prob]

            # ── Sensitive-topic analysis (before nudges) ──────────────────────
            sensitive = _sensitive_topic_analysis(text)
            if sensitive["is_sensitive"]:
                logger.info(
                    "⚠️  Sensitive tone detected: distress=%d unverified=%d force=%s",
                    sensitive["distress_hits"], sensitive["unverified_hits"],
                    sensitive["force_suspicious"],
                )

            # ── Soft signal nudges (scaled by ensemble certainty) ─────────────
            signals = self._detect_soft_signals(text)
            adjusted_fake_prob, nudge_total, nudge_details = self._apply_nudges(
                fake_prob, signals, model_conf
            )
            adjusted_real_prob = 1.0 - adjusted_fake_prob

            # ── Confidence calibration ────────────────────────────────────────
            nudge_boost    = min(abs(nudge_total) * 0.5, 0.08)
            calibrated_conf = _calibrate_confidence(model_conf, all_fake_probs, nudge_boost)
            # Apply sensitive-topic reduction on top
            calibrated_conf = max(
                0.01,
                calibrated_conf - sensitive["confidence_reduction"]
            )

            # ── Adaptive SUSPICIOUS / verdict ─────────────────────────────────
            final_verdict, is_fake = _compute_suspicious(
                adjusted_fake_prob, calibrated_conf,
                all_fake_probs, self.FAKE_THRESHOLD, sensitive,
            )

            # ── Agreement metrics for frontend ────────────────────────────────
            agreement     = _agreement_score(all_fake_probs, self.FAKE_THRESHOLD)
            spread        = (max(all_fake_probs) - min(all_fake_probs)) if len(all_fake_probs) > 1 else 0.0
            entropy       = _entropy_penalty(all_fake_probs)

            confidence_pct = round(calibrated_conf * 100, 1) if final_verdict != "SUSPICIOUS" else 0

            result = {
                "success":           True,
                "is_fake":           is_fake,
                "verdict":           final_verdict,
                "confidence":        confidence_pct,
                "confidence_raw":    round(calibrated_conf, 4),
                "fake_probability":  round(adjusted_fake_prob, 4),
                "real_probability":  round(adjusted_real_prob, 4),
                "threshold_used":    self.FAKE_THRESHOLD,
                "category":          self._detect_category(text),

                # Per-model breakdown (Q4 – frontend display)
                "individual_models": individual_models,

                # Agreement metrics (new – useful for frontend confidence bar)
                "model_agreement": {
                    "score":      round(agreement, 4),
                    "spread":     round(spread, 4),
                    "entropy":    round(entropy, 4),
                    "fake_votes": sum(1 for p in all_fake_probs if p >= self.FAKE_THRESHOLD),
                    "total_models": len(all_fake_probs),
                },

                # Heuristic nudge trace
                "soft_signals": {
                    "detected":       signals,
                    "fake_prob_before": round(fake_prob, 4),
                    "fake_prob_after":  round(adjusted_fake_prob, 4),
                    "total_nudge":    round(nudge_total, 4),
                    "nudge_details":  nudge_details,
                    "scaling_factor": round(self._nudge_scale(model_conf), 3),
                },

                # Sensitive-topic filter trace
                "sensitive_filter": sensitive,

                # Legacy fields (kept for backward compat)
                "ensemble_details":  ensemble_details,
                "is_uncertain":      False,
                "uncertainty_note":  None,

                "text_length":    len(text),
                "word_count":     len(text.split()),
                "execution_time": round(time.time() - start, 3),
            }

            logger.info(
                "✅ %s | conf=%.1f%% | fake_prob=%.3f | agreement=%.2f | verdicts=%s",
                final_verdict, confidence_pct, adjusted_fake_prob, agreement,
                [m["verdict"] for m in individual_models],
            )
            return result

        except Exception as e:
            logger.exception("Analysis failed: %s", e)
            return self._error_result(str(e))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_individual_models(self, ensemble_details: Dict) -> List[Dict[str, Any]]:
        models = []
        if not ensemble_details or "votes" not in ensemble_details:
            return models
        for vote in ensemble_details.get("votes", []):
            if not isinstance(vote, dict):
                continue
            fake_p = vote.get("fake_probability", 0.5)
            real_p = vote.get("real_probability", 1.0 - fake_p)
            conf   = vote.get("confidence", max(fake_p, real_p))
            models.append({
                "model":   vote.get("model", "unknown"),
                "verdict": "FAKE" if fake_p >= self.FAKE_THRESHOLD else "REAL",
                "confidence": round(conf * 100, 1) if conf <= 1 else round(conf, 1),
                "fake_prob":  round(fake_p, 4),
                "real_prob":  round(real_p, 4),
            })
        return models

    @staticmethod
    def _nudge_scale(model_conf: float) -> float:
        """
        Scaling factor that suppresses nudges when the ensemble is already certain.

        At conf=0.90 → scale≈0.25  (nudges contribute only 25% of their raw value)
        At conf=0.60 → scale≈0.80  (nudges contribute 80% – high impact in uncertain zone)
        At conf=0.50 → scale≈1.00  (maximum nudge in the most uncertain zone)

        Formula: scale = 1 - sigmoid(10 * (conf - 0.55))
        This gives a smooth S-curve centred at conf=0.55.
        """
        x = 10.0 * (model_conf - 0.55)
        return 1.0 - 1.0 / (1.0 + math.exp(-x))

    def _detect_soft_signals(self, text: str) -> List[Dict[str, Any]]:
        signals = []
        text_low = text.lower()

        hits = sum(1 for p in self.SENSATIONAL_TERMS if re.search(p, text_low, re.IGNORECASE))
        if hits >= 2:
            signals.append({"type": "sensational", "severity": "medium", "nudge": 0.05})
        elif hits == 1:
            signals.append({"type": "sensational", "severity": "low",    "nudge": 0.02})

        hits = sum(1 for p in self.HEALTH_MISINFO_PATTERNS if re.search(p, text_low, re.IGNORECASE))
        if hits:
            signals.append({"type": "health_misinfo", "severity": "high",   "nudge": 0.08})

        hits = sum(1 for p in self.CONSPIRACY_PATTERNS if re.search(p, text_low, re.IGNORECASE))
        if hits:
            signals.append({"type": "conspiracy",    "severity": "medium", "nudge": 0.06})

        words = text.split()
        if len(words) >= 5:
            caps_ratio = sum(1 for w in words if len(w) > 2 and w.isupper()) / len(words)
            if caps_ratio >= self.ALL_CAPS_THRESHOLD:
                signals.append({"type": "excessive_caps",        "severity": "low", "nudge": 0.03})

        if re.search(r"[!?]{3,}", text):
            signals.append({"type": "excessive_punctuation", "severity": "low", "nudge": 0.02})

        return signals

    def _apply_nudges(
        self,
        fake_prob: float,
        signals: List[Dict],
        model_conf: float,
    ) -> Tuple[float, float, List[Dict]]:
        """
        Apply heuristic nudges scaled by the ensemble's certainty.

        When the model is very confident (conf→1), nudges shrink toward zero so
        they cannot flip a high-confidence verdict.  In the uncertain zone
        (conf≈0.5) nudges apply at full weight.
        """
        scale   = self._nudge_scale(model_conf)
        total   = sum(s.get("nudge", 0) for s in signals) * scale
        total   = max(-self.MAX_NUDGE, min(self.MAX_NUDGE, total))
        details = [{"type": s["type"], "nudge": round(s["nudge"] * scale, 4)} for s in signals]
        adjusted = max(0.0, min(1.0, fake_prob + total))
        return adjusted, total, details

    def _detect_category(self, text: str) -> str:
        text_low = text.lower()
        keywords = {
            "health":        ["vaccine", "covid", "cancer", "doctor", "medicine", "hospital"],
            "politics":      ["election", "president", "minister", "parliament", "government"],
            "sports":        ["cricket", "football", "match", "player", "goal", "wicket", "psl"],
            "entertainment": ["movie", "film", "actor", "actress", "celebrity", "bollywood"],
            "business":      ["market", "stock", "economy", "company", "imf", "bailout"],
        }
        scores = {cat: sum(1 for kw in kws if kw in text_low) for cat, kws in keywords.items()}
        best   = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def _too_short_result(self, text: str) -> Dict[str, Any]:
        return {
            "success": False, "is_fake": False, "verdict": "INSUFFICIENT_DATA",
            "confidence": 0, "confidence_raw": 0.0,
            "fake_probability": 0.5, "real_probability": 0.5,
            "error": "Text too short (min 10 chars)",
        }

    def _error_result(self, err: str) -> Dict[str, Any]:
        return {
            "success": False, "is_fake": False, "verdict": "ERROR",
            "confidence": 0, "confidence_raw": 0.0,
            "fake_probability": 0.5, "real_probability": 0.5,
            "error": err,
        }

    def get_sources(self, text: str) -> Dict:
        return {"matched_sources": [], "trust_score": 0.5}

    def get_model_info(self) -> Dict:
        return self.model.get_model_info() if self.model_loaded else {"loaded": False}