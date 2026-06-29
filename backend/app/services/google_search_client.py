# app/services/google_search_client.py
"""
Multi-tier web search client for the fake-news detector.

Why the old approach (scraping cse.google.com) never works
-----------------------------------------------------------
Google's public CSE page is a JavaScript SPA.  Results are fetched
*after* page load via XHR calls that carry ephemeral tokens, cookies,
and browser-fingerprint signals.  Playwright/cloudscraper get a 200 OK
but an empty DOM because Google's "SearchGuard" (rolled out Jan 2025)
blocks all headless traffic before the XHR fires.  There is no public
JSONP endpoint that bypasses this.

Multi-tier strategy (in order of preference)
--------------------------------------------
Tier 1 – Serper.dev  (RECOMMENDED – returns real Google results fast)
    • Free: 2,500 searches on sign-up, no credit card.
    • Returns Facebook, Instagram, Twitter results reliably.
    • Env var: SERPER_API_KEY
    • Sign up: https://serper.dev

Tier 2 – Google Custom Search JSON API  (official, 100 free/day)
    • Different from the public CSE page – this is the keyed REST API.
    • Needs two env vars: GOOGLE_API_KEY + GOOGLE_CSE_ID
    • Get API key: https://console.cloud.google.com/apis/library/customsearch.googleapis.com
    • Get CSE ID:  https://programmablesearchengine.google.com/
    • Free: 100 queries/day; $5 per 1000 after that.

Tier 3 – Direct Google SERP scrape  (low-volume fallback, ~10 req/day safe)
    • No keys needed.  Uses a "Lynx" browser user-agent trick that
      still works in 2025 for very low traffic.
    • Will occasionally return a CAPTCHA page; code detects this and
      moves on.

Set at least ONE of:
    SERPER_API_KEY          → enables Tier 1 (recommended)
    GOOGLE_API_KEY + GOOGLE_CSE_ID  → enables Tier 2
If neither is set, Tier 3 (keyless scrape) is used as a best-effort.
"""

import os
import re
import json
import time
import random
import logging
from urllib.parse import quote, urlparse, urlencode

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Domain classification sets ────────────────────────────────────────────────

SOCIAL_MEDIA_DOMAINS = {
    'facebook.com', 'fb.com', 'instagram.com', 'twitter.com', 'x.com',
    'reddit.com', 'tiktok.com', 'youtube.com', 'linkedin.com',
    'whatsapp.com', 'telegram.org', 'snapchat.com', 'threads.net',
    'pinterest.com', 'tumblr.com',
}

CREDIBLE_DOMAINS = {
    # International
    'bbc.com', 'bbc.co.uk', 'cnn.com', 'reuters.com', 'apnews.com',
    'aljazeera.com', 'theguardian.com', 'nytimes.com', 'washingtonpost.com',
    'bloomberg.com', 'ft.com', 'economist.com', 'time.com',
    'nbcnews.com', 'abcnews.go.com', 'cbsnews.com', 'npr.org',
    # Pakistani outlets
    'dawn.com', 'geo.tv', 'arynews.tv', 'tribune.com.pk',
    'dunyanews.tv', 'samaa.tv', 'urdupoint.com', 'jang.com.pk',
    'thenews.com.pk', 'express.com.pk', 'brecorder.com',
    # Fact-checkers
    'snopes.com', 'factcheck.org', 'politifact.com', 'fullfact.org',
}


# ── Main client ───────────────────────────────────────────────────────────────

class GoogleSearchClient:
    """
    Multi-tier search client.  Falls through tiers automatically.

    Quick-start:
        # .env or environment
        SERPER_API_KEY=your_key          # get free at https://serper.dev
        # or
        GOOGLE_API_KEY=your_key
        GOOGLE_CSE_ID=your_cx_id
    """

    def __init__(self):
        self.serper_key    = os.environ.get("SERPER_API_KEY", "").strip()
        self.google_key    = os.environ.get("GOOGLE_API_KEY", "").strip()
        self.google_cse_id = os.environ.get("GOOGLE_CSE_ID", "").strip()

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self._random_ua()})

        tiers = []
        if self.serper_key:
            tiers.append("Serper.dev (Tier 1)")
        if self.google_key and self.google_cse_id:
            tiers.append("Google CSE JSON API (Tier 2)")
        tiers.append("Direct SERP scrape (Tier 3, keyless fallback)")

        logger.info(f"🌐 GoogleSearchClient ready – active tiers: {', '.join(tiers)}")

    # ── Public API ────────────────────────────────────────────────────────────

    def search_claim(self, claim: str, num_results: int = 10) -> dict:
        """
        Search for a claim across the web and categorise results.

        Returns:
            {
                "success": bool,
                "tier_used": str,
                "total_results": int,
                "message": str,
                "credible_sources": [...],
                "social_media_sources": [...],
                "other_sources": [...],
            }
        """
        results, tier_used = [], "none"

        # ── Tier 1: Serper.dev ─────────────────────────────────────────────
        if self.serper_key and not results:
            try:
                results = self._serper_search(claim, num_results)
                if results:
                    tier_used = "serper"
                    logger.info(f"✅ Tier 1 (Serper) returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Tier 1 (Serper) failed: {e}")

        # ── Tier 2: Google Custom Search JSON API ──────────────────────────
        if self.google_key and self.google_cse_id and not results:
            try:
                results = self._google_cse_api(claim, num_results)
                if results:
                    tier_used = "google_cse_api"
                    logger.info(f"✅ Tier 2 (Google CSE API) returned {len(results)} results")
            except Exception as e:
                logger.warning(f"Tier 2 (Google CSE API) failed: {e}")

        # ── Tier 3: Direct Google SERP scrape (keyless) ────────────────────
        if not results:
            try:
                results = self._direct_google_scrape(claim, num_results)
                if results:
                    tier_used = "direct_scrape"
                    logger.info(f"✅ Tier 3 (direct scrape) returned {len(results)} results")
                else:
                    logger.warning("Tier 3 (direct scrape) returned 0 results – possible CAPTCHA")
            except Exception as e:
                logger.warning(f"Tier 3 (direct scrape) failed: {e}")

        if not results:
            return {
                "success": False,
                "error": (
                    "All search tiers failed.  Set SERPER_API_KEY (free at serper.dev) "
                    "or GOOGLE_API_KEY + GOOGLE_CSE_ID to enable reliable results."
                ),
            }

        credible, social, other = self._categorise(results[:num_results])
        return {
            "success": True,
            "tier_used": tier_used,
            "total_results": len(results),
            "message": self._build_message(credible, social),
            "credible_sources": credible,
            "social_media_sources": social,
            "other_sources": other,
        }

    # ── Tier 1: Serper.dev ────────────────────────────────────────────────────

    def _serper_search(self, query: str, num: int) -> list[dict]:
        """
        Call the Serper.dev Google Search API.
        Docs: https://serper.dev/

        Returns real Google results including social media pages.
        The free tier gives 2,500 searches – more than enough for a
        fact-checking tool.  No credit card required at sign-up.
        """
        url = "https://google.serper.dev/search"
        payload = {
            "q": query,
            "num": min(num, 10),   # Serper max per call is 10
            "gl": "us",             # geolocation for broader results
            "hl": "en",
        }
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json",
        }
        resp = self.session.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        # Organic results (news, blog posts, social media pages)
        for item in data.get("organic", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        # Top stories (news cards)
        for item in data.get("topStories", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("date", ""),   # date is best snippet for news cards
            })
        return results

    # ── Tier 2: Official Google Custom Search JSON API ────────────────────────

    def _google_cse_api(self, query: str, num: int) -> list[dict]:
        """
        Call the official Google Custom Search JSON API.

        This is NOT the public CSE page – it's the authenticated REST
        endpoint that actually works from Python.

        Free: 100 queries/day.
        Endpoint: https://www.googleapis.com/customsearch/v1

        To get credentials:
          1. Enable "Custom Search API" at https://console.cloud.google.com
          2. Create an API key (restrict it to Custom Search API for safety)
          3. Create a search engine at https://programmablesearchengine.google.com/
             → set it to "Search the entire web"
             → copy the Search Engine ID (cx)
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.google_key,
            "cx":  self.google_cse_id,
            "q":   query,
            "num": min(num, 10),   # API max is 10 per page
        }
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("items", []):
            results.append({
                "title":   item.get("title", ""),
                "url":     item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })
        return results

    # ── Tier 3: Direct Google SERP scrape (keyless) ───────────────────────────

    def _direct_google_scrape(self, query: str, num: int) -> list[dict]:
        """
        Scrape Google search results directly – no keys needed.

        Uses a text-browser User-Agent (Lynx) that historically receives
        a simpler HTML page from Google rather than a CAPTCHA redirect.
        Works reliably at very low volumes (< ~15 requests/day per IP).

        If Google returns a CAPTCHA page we return [] so the caller can
        handle the failure gracefully.
        """
        params = {
            "q":    query,
            "num":  min(num, 10),
            "hl":   "en",
            "gl":   "us",
            "safe": "off",
        }
        url = "https://www.google.com/search?" + urlencode(params)

        # Lynx UA gets a simpler, less bot-resistant page from Google.
        headers = {
            "User-Agent": (
                "Lynx/2.8.9rel.1 libwww-FM/2.14 SSL-MM/1.4.1 OpenSSL/1.0.2u"
            ),
            "Accept":          "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection":      "keep-alive",
        }

        time.sleep(random.uniform(1.5, 3.0))   # polite delay
        resp = self.session.get(url, headers=headers, timeout=20, allow_redirects=True)

        if resp.status_code != 200:
            logger.warning(f"Direct scrape HTTP {resp.status_code}")
            return []

        html = resp.text

        # Detect CAPTCHA / unusual traffic page
        if any(marker in html for marker in (
            "detected unusual traffic",
            "recaptcha",
            "CAPTCHA",
            "/sorry/index",
        )):
            logger.warning("Direct scrape: Google CAPTCHA detected – skipping tier 3")
            return []

        return self._parse_google_html(html)

    def _parse_google_html(self, html: str) -> list[dict]:
        """Parse standard Google SERP HTML into result dicts."""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        # Primary selector for modern Google HTML layout
        for div in soup.select("div.g"):
            title_el   = div.select_one("h3")
            anchor_el  = div.select_one("a[href]")
            snippet_el = div.select_one("div.VwiC3b, div[data-sncf], span.aCOpRe")

            href = anchor_el["href"] if anchor_el else ""
            # Google wraps URLs in /url?q=... strip that
            href = re.sub(r"^/url\?q=([^&]+).*", lambda m: m.group(1), href)

            if href.startswith("http") and title_el:
                results.append({
                    "title":   title_el.get_text(strip=True),
                    "url":     href,
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })

        # Fallback: simpler layout returned for Lynx UA
        if not results:
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if href.startswith("http") and not any(
                    x in href for x in ("google.com", "googleapis.com", "gstatic.com")
                ):
                    results.append({
                        "title":   a.get_text(strip=True) or href,
                        "url":     href,
                        "snippet": "",
                    })

        return results

    # ── Categorisation helpers ─────────────────────────────────────────────────

    def _categorise(self, results: list[dict]):
        credible, social, other = [], [], []
        for r in results:
            domain = self._extract_domain(r.get("url", ""))
            if self._is_social(domain):
                social.append(r)
            elif self._is_credible(domain):
                credible.append(r)
            else:
                other.append(r)
        return credible, social, other

    def _extract_domain(self, url: str) -> str:
        try:
            netloc = urlparse(url).netloc.lower()
            return netloc.lstrip("www.")
        except Exception:
            return ""

    def _is_social(self, domain: str) -> bool:
        return any(sm in domain for sm in SOCIAL_MEDIA_DOMAINS)

    def _is_credible(self, domain: str) -> bool:
        return any(pat in domain for pat in CREDIBLE_DOMAINS)

    def _build_message(self, credible: list, social: list) -> str:
        if social and not credible:
            return (
                "⚠️ This claim circulates only on social media platforms. "
                "No authoritative news source has reported it."
            )
        if credible and social:
            return (
                f"📰 Covered by {len(credible)} credible outlet(s) and "
                f"discussed on {len(social)} social platform(s)."
            )
        if credible:
            return f"✅ Found coverage on {len(credible)} credible news outlet(s)."
        return "⚠️ Limited or no verified web presence found for this claim."

    # ── Misc helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _random_ua() -> str:
        uas = [
            (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/17.0 Safari/605.1.15"
            ),
            (
                "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) "
                "Gecko/20100101 Firefox/126.0"
            ),
        ]
        return random.choice(uas)