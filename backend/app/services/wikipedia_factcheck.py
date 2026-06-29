# app/services/wikipedia_factcheck.py
"""
TruthLens Wikipedia Fact‑Check Module
=====================================
Searches Wikipedia for a claim and returns whether the claim was
verified, a related article URL, and a human‑readable verdict.
"""

import logging
import requests
from typing import Dict, Any

logger = logging.getLogger(__name__)

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"


class WikipediaFactCheckModule:
    """
    Simple Wikipedia fact‑checking module.
    Searches for the claim text and returns the best matching article.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "TruthLens/1.0 (educational project; contact@truthlens.ai)"
        })
        logger.info("📚 Wikipedia Fact-Checker initialized")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze a text claim against Wikipedia.

        Returns:
            {
                "verified": bool,
                "verdict": str,        # e.g. "VERIFIED", "NOT_FOUND", "RELATED_ARTICLE"
                "confidence": float,   # 0.0 - 1.0
                "wikipedia_url": str,  # full URL of related article
                "article_title": str,
                "snippet": str,
                "error": str (if any)
            }
        """
        if not text or len(text.strip()) < 10:
            return {
                "verified": False,
                "verdict": "INSUFFICIENT_DATA",
                "confidence": 0.0,
                "wikipedia_url": "",
                "article_title": "",
                "snippet": "",
                "error": "Text too short",
            }

        query = text.strip()
        try:
            # Search Wikipedia
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            }
            resp = self.session.get(WIKIPEDIA_API, params=search_params, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            search_results = data.get("query", {}).get("search", [])
            if not search_results:
                return {
                    "verified": False,
                    "verdict": "NOT_FOUND",
                    "confidence": 0.0,
                    "wikipedia_url": "",
                    "article_title": "",
                    "snippet": "",
                }

            # Pick the first (most relevant) result
            best = search_results[0]
            title = best.get("title", "")
            snippet = best.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
            page_id = best.get("pageid")

            wikipedia_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"

            # Attempt to determine if the article actually supports the claim.
            # For simplicity, we mark it as "VERIFIED" if the title contains
            # one of the claim's main keywords, else "RELATED_ARTICLE".
            # A more advanced version could use NLP similarity.
            claim_keywords = [w for w in query.lower().split() if len(w) > 3]
            title_matches = sum(1 for kw in claim_keywords if kw in title.lower())
            verified = title_matches >= 2

            verdict = "VERIFIED" if verified else "RELATED_ARTICLE"
            confidence = 0.8 if verified else 0.5

            return {
                "verified": verified,
                "verdict": verdict,
                "confidence": confidence,
                "wikipedia_url": wikipedia_url,
                "article_title": title,
                "snippet": snippet,
                "page_id": page_id,
            }

        except Exception as e:
            logger.error(f"Wikipedia fact-check failed: {e}")
            return {
                "verified": False,
                "verdict": "ERROR",
                "confidence": 0.0,
                "wikipedia_url": "",
                "article_title": "",
                "snippet": "",
                "error": str(e),
            }