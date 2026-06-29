
"""
app/services/url_ml_detector.py

Hybrid URL Malicious/Phishing Detector – 3 layers (GSB + URLBERT + heuristics)
Plus dynamic WHOIS reputation override.

Fixes: Legitimate domains with low heuristic score (<20) and ML-only false positive
are now demoted to CLEAN instead of SUSPICIOUS.
"""

import os
import re
import time
import json
import hashlib
import logging
import ipaddress
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️ transformers not installed – URLBERT disabled, using heuristics only")

try:
    import whois
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False
    logger.warning("⚠️ python-whois not installed – WHOIS reputation disabled")


MALICIOUS_MODEL = "CrabInHoney/urlbert-tiny-v4-malicious-url-classifier"
PHISHING_MODEL  = "CrabInHoney/urlbert-tiny-v4-phishing-classifier"

MALICIOUS_LABELS = {
    "LABEL_0": "benign",
    "LABEL_1": "defacement",
    "LABEL_2": "malware",
    "LABEL_3": "phishing",
}
PHISHING_LABELS = {
    "LABEL_0": "good",
    "LABEL_1": "phishing",
}

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club", ".online",
    ".info", ".biz", ".work", ".loan", ".click", ".link", ".download",
    ".zip", ".mov"
}

SUSPICIOUS_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm",
    "banking", "paypal", "amazon", "microsoft", "apple", "google", "facebook",
    "password", "credential", "wallet", "suspend", "unusual", "activity",
    "validate", "authenticate"
]

CACHE_TTL_SECONDS = 3600

# Keywords used in WHOIS organisation/registrant fields to identify legitimate corporations
TRUSTED_ORG_KEYWORDS = [
    "google", "microsoft", "apple", "amazon", "facebook", "meta",
    "netflix", "twitter", "linkedin", "github", "stackoverflow"
]


class URLMLDetector:
    def __init__(self):
        self._gsb_key = os.environ.get("GOOGLE_SAFE_BROWSING_KEY", "")
        self._mal_classifier = None
        self._phi_classifier = None
        self._models_loaded = False
        self._models_attempted = False
        self._cache = {}
        logger.info(
            f"🔗 URLMLDetector init | GSB={'enabled' if self._gsb_key else 'no key'} | "
            f"URLBERT={'available' if TRANSFORMERS_AVAILABLE else 'unavailable'} | "
            f"WHOIS={'available' if WHOIS_AVAILABLE else 'unavailable'}"
        )

    def load(self) -> bool:
        """Pre-load URLBERT models (optional, called at startup)."""
        return self._load_models()

    def analyze(self, url: str) -> dict:
        if not url or not isinstance(url, str):
            return self._error_response("No URL provided")
        url = url.strip()

        cached = self._from_cache(url)
        if cached:
            cached["from_cache"] = True
            return cached

        details = {}
        methods_used = []

        # Extract domain for later use
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
            domain = parsed.netloc or ""
            main_domain = self._extract_main_domain(domain)
        except Exception:
            main_domain = ""

        # ---- WHOIS reputation check (dynamic, no hardcoded domain list) ----
        if WHOIS_AVAILABLE and main_domain:
            try:
                w = whois.whois(main_domain)
                registrant = (w.org or w.registrant or "").lower()
                if any(kw in registrant for kw in TRUSTED_ORG_KEYWORDS):
                    result = self._result(
                        url, "CLEAN", 0.99, 5, "WHOIS_reputation",
                        {}, "Registered to known legitimate organization"
                    )
                    self._to_cache(url, result)
                    return result
            except Exception as e:
                logger.debug(f"WHOIS lookup failed for {main_domain}: {e}")

        # ---- Layer 1: Google Safe Browsing ----
        if self._gsb_key:
            gsb = self._check_gsb(url)
            details["google_safe_browsing"] = gsb
            if gsb["verdict"] == "MALICIOUS":
                result = self._result(
                    url, "MALICIOUS", gsb["confidence"],
                    method="Google Safe Browsing",
                    classification=gsb.get("threat_type", "malware"),
                    message=f"Flagged by Google Safe Browsing: {gsb.get('threat_type', 'MALICIOUS')}"
                )
                self._to_cache(url, result)
                return result
            methods_used.append("Google Safe Browsing")

        # ---- Layer 2: URLBERT dual models ----
        if not self._models_attempted:
            self._load_models()
        if self._models_loaded:
            ml_result = self._check_urlbert(url)
            details["urlbert"] = ml_result
            methods_used.append(f"URLBERT ({ml_result.get('classification', '?')})")
        else:
            ml_result = {"verdict": "UNKNOWN", "confidence": 0.0}

        # ---- Layer 3: Heuristics (with domain-name filtering) ----
        h_result = self._check_heuristics(url, main_domain)
        details["heuristics"] = h_result
        methods_used.append(f"Heuristics (score={h_result['raw_score']})")

        # ---- Combine verdicts (fixed) ----
        final_verdict, final_conf = self._combine(ml_result, h_result)

        # Convert to orchestrator‑expected fields
        malicious = 1 if final_verdict == "MALICIOUS" else 0
        suspicious = 1 if final_verdict == "SUSPICIOUS" else 0
        score = self._verdict_to_score(final_verdict, final_conf)
        method = " + ".join(methods_used) if methods_used else "Heuristics"
        classification = self._get_dominant_class(details)
        message = self._build_message(final_verdict, details)

        result = self._result(url, final_verdict, final_conf, score, method,
                              classification, message, details)
        self._to_cache(url, result)
        return result

    # ================= Helper methods =================

    def _extract_main_domain(self, host: str) -> str:
        """Extract second-level domain (e.g., 'google' from 'www.google.com')."""
        parts = host.rstrip('.').split('.')
        if len(parts) >= 2:
            return parts[-2].lower()
        return parts[0].lower() if parts else ""

    def _check_gsb(self, url: str) -> dict:
        endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={self._gsb_key}"
        payload = {
            "client": {"clientId": "truthlens", "clientVersion": "1.0"},
            "threatInfo": {
                "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING",
                                "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
                "platformTypes": ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries": [{"url": url}],
            },
        }
        try:
            req = urllib.request.Request(
                endpoint, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode())
                matches = body.get("matches", [])
                if matches:
                    threat_type = matches[0].get("threatType", "MALICIOUS")
                    return {"verdict": "MALICIOUS", "threat_type": threat_type, "confidence": 0.99}
                return {"verdict": "CLEAN", "confidence": 0.95}
        except Exception as e:
            logger.debug(f"GSB error: {e}")
            return {"verdict": "UNKNOWN", "error": str(e)}

    def _load_models(self) -> bool:
        self._models_attempted = True
        if not TRANSFORMERS_AVAILABLE:
            return False
        try:
            device = 0 if torch.cuda.is_available() else -1
            logger.info("⏳ Loading URLBERT models...")
            self._mal_classifier = pipeline(
                "text-classification",
                model=MALICIOUS_MODEL,
                tokenizer=MALICIOUS_MODEL,
                return_all_scores=True,
                device=device,
            )
            self._phi_classifier = pipeline(
                "text-classification",
                model=PHISHING_MODEL,
                tokenizer=PHISHING_MODEL,
                return_all_scores=True,
                device=device,
            )
            self._mal_classifier("https://google.com")
            self._phi_classifier("https://google.com")
            self._models_loaded = True
            logger.info("✅ URLBERT dual models loaded")
            return True
        except Exception as e:
            logger.warning(f"⚠️ URLBERT load failed: {e} — using heuristics only")
            self._models_loaded = False
            return False

    def _check_urlbert(self, url: str) -> dict:
        try:
            scores_a = {
                MALICIOUS_LABELS[r["label"]]: r["score"]
                for r in self._mal_classifier(url[:512])[0]
            }
            benign_a = scores_a.get("benign", 0.0)
            malware_a = scores_a.get("malware", 0.0)
            deface_a = scores_a.get("defacement", 0.0)
            phishing_a = scores_a.get("phishing", 0.0)

            scores_b = {
                PHISHING_LABELS[r["label"]]: r["score"]
                for r in self._phi_classifier(url[:512])[0]
            }
            good_b = scores_b.get("good", 0.0)
            phishing_b = scores_b.get("phishing", 0.0)

            phishing_combined = max(phishing_a, phishing_b)
            malware_combined = max(malware_a, deface_a)
            malicious_prob = max(phishing_combined, malware_combined)
            benign_prob = benign_a * good_b

            if malicious_prob >= 0.70:
                verdict = "MALICIOUS"
                confidence = malicious_prob
            elif malicious_prob >= 0.40:
                verdict = "SUSPICIOUS"
                confidence = malicious_prob
            else:
                verdict = "CLEAN"
                confidence = benign_prob

            classification = "benign"
            if phishing_combined >= malware_combined and phishing_combined >= 0.40:
                classification = "phishing"
            elif malware_a >= 0.40:
                classification = "malware"
            elif deface_a >= 0.40:
                classification = "defacement"

            return {
                "verdict": verdict,
                "confidence": round(confidence, 4),
                "classification": classification,
                "scores": {
                    "benign": round(benign_a, 4),
                    "malware": round(malware_a, 4),
                    "defacement": round(deface_a, 4),
                    "phishing_a": round(phishing_a, 4),
                    "phishing_b": round(phishing_b, 4),
                }
            }
        except Exception as e:
            logger.debug(f"URLBERT error: {e}")
            return {"verdict": "UNKNOWN", "confidence": 0.0}

    def _check_heuristics(self, url: str, main_domain: str) -> dict:
        flags = []
        score = 0
        try:
            parsed = urllib.parse.urlparse(url if "://" in url else f"http://{url}")
        except Exception:
            return {"verdict": "UNKNOWN", "raw_score": 0, "flags": ["unparseable"]}
        host = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        full = url.lower()

        # ── Length ─────────────────────────────────────────
        if len(url) > 200:
            score += 20
            flags.append(f"very long URL ({len(url)} chars)")
        elif len(url) > 100:
            score += 10
            flags.append(f"long URL ({len(url)} chars)")

        # ── IP address as host ────────────────────────────
        try:
            ipaddress.ip_address(host)
            score += 30
            flags.append("IP address used as hostname")
        except ValueError:
            pass

        # ── Excessive subdomains ──────────────────────────
        parts = host.split(".")
        if len(parts) > 5:
            score += 15
            flags.append(f"excessive subdomains ({len(parts)} levels)")

        # ── Suspicious TLD ────────────────────────────────
        tld = "." + parts[-1] if parts else ""
        if tld in SUSPICIOUS_TLDS:
            score += 20
            flags.append(f"suspicious TLD ({tld})")

        # ── Brand in subdomain but different root ─────────
        brands = ["paypal", "amazon", "google", "facebook", "microsoft",
                  "apple", "netflix", "instagram", "twitter"]
        root = ".".join(parts[-2:]) if len(parts) >= 2 else host
        for brand in brands:
            if brand in host and brand not in root:
                score += 35
                flags.append(f"brand '{brand}' in subdomain, not in root")
                break

        # ── Phishing keywords (excluding domain's own name) ──
        kw_hits = [kw for kw in SUSPICIOUS_KEYWORDS if kw in full]
        # Remove keyword if it matches the main domain (e.g., "google" in google.com)
        kw_hits = [kw for kw in kw_hits if kw != main_domain]

        if len(kw_hits) >= 3:
            score += 25
            flags.append(f"multiple phishing keywords: {kw_hits[:5]}")
        elif kw_hits:
            score += 10
            flags.append(f"phishing keyword(s): {kw_hits[:3]}")

        # ── Non‑standard port ────────────────────────────
        if parsed.port and parsed.port not in (80, 443, 8080, 8443):
            score += 15
            flags.append(f"non‑standard port ({parsed.port})")

        # ── Special characters in path ────────────────────
        special = sum(1 for c in path + query if c in r"=-_~!@#$%^&*()[]{}|;:,<>?")
        if special > 15:
            score += 10
            flags.append(f"high special‑char density ({special})")

        # ── Percent‑encoding in host ─────────────────────
        if "%" in host:
            score += 20
            flags.append("percent‑encoding in hostname")

        # ── Multiple redirectors ─────────────────────────
        redirect_count = full.count("http://") + full.count("https://")
        if redirect_count > 1:
            score += 25
            flags.append("URL contains embedded redirect")

        # ── HTTPS missing ────────────────────────────────
        if parsed.scheme != "https":
            score += 10
            flags.append("not using HTTPS")

        # ── Verdict ──────────────────────────────────────
        if score >= 60:
            verdict = "MALICIOUS"
            confidence = min(0.95, 0.60 + (score - 60) / 200)
        elif score >= 30:
            verdict = "SUSPICIOUS"
            confidence = min(0.80, 0.40 + (score - 30) / 150)
        else:
            verdict = "CLEAN"
            confidence = max(0.5, 1.0 - score / 60)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "raw_score": score,
            "flags": flags,
        }

    def _combine(self, ml: dict, heur: dict) -> tuple:
        """
        Combine URLBERT and heuristic verdicts.
        Priority:
          1. If heuristics says CLEAN with very low raw_score (<20) and ML says MALICIOUS → CLEAN.
          2. Else if heuristics CLEAN with raw_score<30 and ML MALICIOUS → SUSPICIOUS.
          3. Otherwise standard weighted merging.
        """
        ml_verdict = ml.get("verdict", "UNKNOWN")
        h_verdict = heur.get("verdict", "CLEAN")
        ml_conf = ml.get("confidence", 0.0)
        h_conf = heur.get("confidence", 0.5)

        # NEW: If heuristics is CLEAN with raw_score <20 and ML says MALICIOUS -> trust heuristics
        if h_verdict == "CLEAN" and heur.get("raw_score", 0) < 20 and ml_verdict == "MALICIOUS":
            return "CLEAN", max(h_conf, 0.8)

        # Existing downdgrade rule (raw_score<30 -> SUSPICIOUS)
        if heur.get("raw_score", 0) < 30 and h_verdict == "CLEAN" and ml_verdict == "MALICIOUS":
            return "SUSPICIOUS", min(max(ml_conf * 0.4, 0.3), 0.7)

        if ml_verdict == h_verdict and ml_verdict != "UNKNOWN":
            combined_conf = ml_conf * 0.65 + h_conf * 0.35
            return ml_verdict, round(combined_conf, 4)

        if ml_verdict == "MALICIOUS" or h_verdict == "MALICIOUS":
            conf = max(ml_conf if ml_verdict == "MALICIOUS" else 0,
                       h_conf if h_verdict == "MALICIOUS" else 0)
            return "MALICIOUS", round(conf, 4)

        if ml_verdict == "SUSPICIOUS" or h_verdict == "SUSPICIOUS":
            conf = max(ml_conf if ml_verdict == "SUSPICIOUS" else 0,
                       h_conf if h_verdict == "SUSPICIOUS" else 0)
            return "SUSPICIOUS", round(conf, 4)

        if ml_verdict == "UNKNOWN":
            return h_verdict, h_conf

        return "CLEAN", max(ml_conf, h_conf)

    @staticmethod
    def _verdict_to_score(verdict: str, confidence: float) -> int:
        base = {"MALICIOUS": 80, "SUSPICIOUS": 45, "CLEAN": 5, "UNKNOWN": 10}
        return min(100, int(base.get(verdict, 10) + confidence * 15))

    def _get_dominant_class(self, details: dict) -> str:
        if "urlbert" in details and "classification" in details["urlbert"]:
            return details["urlbert"]["classification"]
        if "heuristics" in details and details["heuristics"].get("flags"):
            return "heuristic"
        return "unknown"

    @staticmethod
    def _build_message(verdict: str, details: dict) -> str:
        parts = []
        if "google_safe_browsing" in details:
            gsb = details["google_safe_browsing"]
            if gsb.get("verdict") == "MALICIOUS":
                parts.append(f"Google Safe Browsing: {gsb.get('threat_type', 'THREAT')}")
        if "urlbert" in details:
            ub = details["urlbert"]
            if ub.get("verdict") in ("MALICIOUS", "SUSPICIOUS"):
                parts.append(f"URLBERT: {ub.get('classification', ub['verdict'])}")
        if "heuristics" in details:
            h = details["heuristics"]
            if h.get("flags"):
                parts.append(f"Heuristics: {'; '.join(h['flags'][:3])}")
        if not parts:
            return f"URL appears {verdict.lower()}"
        return " | ".join(parts)

    # ---- Cache ----
    def _cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _from_cache(self, url: str) -> Optional[dict]:
        key = self._cache_key(url)
        entry = self._cache.get(key)
        if entry:
            result, ts = entry
            if time.time() - ts < CACHE_TTL_SECONDS:
                result["from_cache"] = True
                return result
            del self._cache[key]
        return None

    def _to_cache(self, url: str, result: dict) -> None:
        to_store = {k: v for k, v in result.items() if k != "from_cache"}
        self._cache[self._cache_key(url)] = (to_store, time.time())

    # ---- Result builders ----
    def _result(self, url, verdict, confidence, score, method,
                classification="", message="", details=None) -> dict:
        return {
            "success": True,
            "url": url,
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "score": score,
            "malicious": 1 if verdict == "MALICIOUS" else 0,
            "suspicious": 1 if verdict == "SUSPICIOUS" else 0,
            "method": method,
            "classification": classification,
            "message": message,
            "details": details or {},
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "from_cache": False,
        }

    def _error_response(self, error: str) -> dict:
        return {
            "success": False,
            "error": error,
            "malicious": 0,
            "suspicious": 0,
            "score": 0,
            "verdict": "UNKNOWN",
        }

    # ---- Compatibility ----
    def get_model_info(self) -> dict:
        return {
            "model_name": "hybrid (GSB + URLBERT v4 + heuristics + WHOIS)",
            "models_loaded": self._models_loaded,
            "gsb_enabled": bool(self._gsb_key),
            "whois_enabled": WHOIS_AVAILABLE,
        }
# app/services/virustotal_client.py
import os
import requests
import hashlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class VirusTotalClient:
    """VirusTotal API client for URL and file reputation checks"""
    
    def __init__(self):
        # Read API key directly from environment at initialization time
        self.api_key = os.environ.get('VIRUSTOTAL_API_KEY')
        self.base_url = 'https://www.virustotal.com/api/v3'
        self.session = requests.Session()
        
        if not self.api_key:
            logger.warning("⚠️ VIRUSTOTAL_API_KEY not set. VirusTotal features disabled.")
            logger.warning("   Please add VIRUSTOTAL_API_KEY to your .env file")
        else:
            self.session.headers.update({'x-apikey': self.api_key})
            logger.info(f"✅ VirusTotal API key loaded (starts with: {self.api_key[:8]}...)")
    
    def _is_configured(self) -> bool:
        """Check if API key is configured"""
        return bool(self.api_key)
    
    def check_url(self, url: str) -> Dict[str, Any]:
        """Check URL reputation"""
        if not self._is_configured():
            return {
                'success': False, 
                'error': 'VirusTotal API key not configured',
                'malicious': 0,
                'suspicious': 0,
                'score': 0,
                'verdict': 'UNKNOWN'
            }
        
        try:
            # Submit URL for analysis
            response = self.session.post(
                f"{self.base_url}/urls",
                data={'url': url}
            )
            
            if response.status_code != 200:
                return {'success': False, 'error': f'API error: {response.status_code}'}
            
            url_id = response.json().get('data', {}).get('id')
            
            if not url_id:
                return {'success': False, 'error': 'No URL ID returned'}
            
            # Get analysis report
            report = self.session.get(f"{self.base_url}/analyses/{url_id}")
            
            if report.status_code == 200:
                data = report.json()
                stats = data.get('data', {}).get('attributes', {}).get('stats', {})
                
                malicious = stats.get('malicious', 0)
                suspicious = stats.get('suspicious', 0)
                
                return {
                    'success': True,
                    'malicious': malicious,
                    'suspicious': suspicious,
                    'score': min((malicious + suspicious) * 2, 100),
                    'verdict': 'MALICIOUS' if malicious > 0 else 'SUSPICIOUS' if suspicious > 0 else 'CLEAN',
                    'stats': stats
                }
            
            return {'success': False, 'error': 'Failed to get report'}
            
        except Exception as e:
            logger.error(f"VirusTotal URL check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def check_file(self, file_path: str) -> Dict[str, Any]:
        """Check file reputation by hash"""
        if not self._is_configured():
            return {
                'success': False, 
                'error': 'VirusTotal API key not configured',
                'malicious': 0,
                'suspicious': 0,
                'score': 0,
                'verdict': 'UNKNOWN'
            }
        
        try:
            # Calculate SHA256
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            
            # Check file report
            response = self.session.get(f"{self.base_url}/files/{file_hash}")
            
            if response.status_code == 200:
                data = response.json()
                attributes = data.get('data', {}).get('attributes', {})
                last_analysis_stats = attributes.get('last_analysis_stats', {})
                
                malicious = last_analysis_stats.get('malicious', 0)
                suspicious = last_analysis_stats.get('suspicious', 0)
                
                return {
                    'success': True,
                    'malicious': malicious,
                    'suspicious': suspicious,
                    'score': min((malicious + suspicious) * 2, 100),
                    'verdict': 'MALICIOUS' if malicious > 0 else 'SUSPICIOUS' if suspicious > 0 else 'CLEAN',
                    'hash': file_hash,
                    'stats': last_analysis_stats
                }
            elif response.status_code == 404:
                return {
                    'success': False, 
                    'error': 'File not found in VirusTotal database', 
                    'hash': file_hash,
                    'malicious': 0,
                    'suspicious': 0,
                    'score': 0,
                    'verdict': 'UNKNOWN'
                }
            else:
                return {'success': False, 'error': f'API error: {response.status_code}'}
            
        except Exception as e:
            logger.error(f"VirusTotal file check failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_file_behavior(self, file_hash: str) -> Dict[str, Any]:
        """Get behavioral analysis from VirusTotal sandbox"""
        if not self._is_configured():
            return {'success': False, 'error': 'API key not configured'}
        
        try:
            response = self.session.get(
                f"{self.base_url}/files/{file_hash}/behaviours"
            )
            
            if response.status_code == 200:
                data = response.json()
                behaviours = data.get('data', [])
                
                parsed = []
                for behaviour in behaviours[:10]:
                    attrs = behaviour.get('attributes', {})
                    parsed.append({
                        'name': attrs.get('name', 'unknown'),
                        'description': attrs.get('description', ''),
                        'severity': attrs.get('severity', 'medium')
                    })
                
                return {
                    'success': True,
                    'behaviors': parsed,
                    'sandbox_verdict': data.get('meta', {}).get('verdict', {})
                }
            
            return {'success': False, 'error': 'No behavior data'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}