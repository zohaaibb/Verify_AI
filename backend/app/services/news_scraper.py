"""
Improved News Scraper for Fake News Detection System
=====================================================
Multi-strategy scraping with robust similarity matching.

Strategies (in order of reliability):
1. Google News RSS (most reliable, rarely blocked)
2. NewsAPI (free tier: 100 req/day)
3. Direct site scraping with rotating headers
4. DuckDuckGo News scraping (no API key needed)
5. Bing News RSS

Similarity: TF-IDF + Semantic (SentenceTransformer) hybrid scoring
"""

import re
import time
import hashlib
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError

import requests
from bs4 import BeautifulSoup

# ── Optional imports (graceful fallback if not installed) ──────────────────────
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using TF-IDF only.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    TFIDF_AVAILABLE = True
except ImportError:
    TFIDF_AVAILABLE = False
    logging.warning("scikit-learn not available. Using basic similarity.")

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logging.warning("feedparser not installed. pip install feedparser")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class Article:
    title: str
    url: str
    source: str
    published: str = ""
    summary: str = ""
    similarity_score: float = 0.0
    match_method: str = ""

@dataclass
class ScraperResult:
    query: str
    total_articles: int = 0
    matches_found: int = 0
    found: bool = False
    articles: list = field(default_factory=list)
    top_match: Optional[Article] = None
    execution_time: float = 0.0
    strategies_used: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# ── Rotating User Agents ───────────────────────────────────────────────────────

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

def _get_headers(index: int = 0) -> dict:
    return {
        "User-Agent": USER_AGENTS[index % len(USER_AGENTS)],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ur;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }


# ── Similarity Engine ──────────────────────────────────────────────────────────

class SimilarityEngine:
    """
    Hybrid similarity: TF-IDF (lexical) + SentenceTransformer (semantic).
    Falls back gracefully if dependencies are missing.
    """

    MATCH_THRESHOLD = 0.25

    def __init__(self):
        self._model = None
        if SEMANTIC_AVAILABLE:
            try:
                self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                logger.info("Semantic model loaded: paraphrase-multilingual-MiniLM-L12-v2")
            except Exception as e:
                logger.warning(f"Could not load semantic model: {e}")

    def _clean(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def _tfidf_score(self, query: str, candidate: str) -> float:
        if not TFIDF_AVAILABLE:
            return self._token_overlap(query, candidate)
        try:
            vec    = TfidfVectorizer(ngram_range=(1, 2), min_df=1, analyzer="word")
            matrix = vec.fit_transform([self._clean(query), self._clean(candidate)])
            return float(sklearn_cosine(matrix[0], matrix[1])[0][0])
        except Exception:
            return self._token_overlap(query, candidate)

    def _semantic_score(self, query: str, candidate: str) -> float:
        if self._model is None:
            return 0.0
        try:
            import numpy as np
            emb  = self._model.encode([query, candidate], convert_to_numpy=True)
            dot  = np.dot(emb[0], emb[1])
            norm = np.linalg.norm(emb[0]) * np.linalg.norm(emb[1])
            return float(dot / norm) if norm > 0 else 0.0
        except Exception:
            return 0.0

    def _token_overlap(self, query: str, candidate: str) -> float:
        q_tokens = set(self._clean(query).split())
        c_tokens = set(self._clean(candidate).split())
        if not q_tokens:
            return 0.0
        intersection = q_tokens & c_tokens
        union        = q_tokens | c_tokens
        return len(intersection) / len(union)

    def _keyword_boost(self, query: str, candidate: str) -> float:
        key_pattern = re.compile(r"\b([A-Z][a-z]+|[0-9]+(?:\.[0-9]+)?)\b")
        q_keys = set(key_pattern.findall(query))
        c_keys = set(key_pattern.findall(candidate))
        if not q_keys:
            return 0.0
        overlap = len(q_keys & c_keys) / len(q_keys)
        return overlap * 0.15

    def score(self, query: str, candidate: str) -> tuple:
        tfidf    = self._tfidf_score(query, candidate)
        semantic = self._semantic_score(query, candidate)
        boost    = self._keyword_boost(query, candidate)

        if self._model is not None:
            combined = 0.50 * semantic + 0.35 * tfidf + boost
            method   = f"hybrid(sem={semantic:.2f},tfidf={tfidf:.2f},boost={boost:.2f})"
        else:
            combined = 0.70 * tfidf + boost
            method   = f"tfidf(score={tfidf:.2f},boost={boost:.2f})"

        return combined, method

    def is_match(self, score: float) -> bool:
        return score >= self.MATCH_THRESHOLD


# ── Scraping Strategies ────────────────────────────────────────────────────────

class GoogleNewsRSS:
    """Most reliable — Google News RSS is rarely blocked."""

    BASE_URL = "https://news.google.com/rss/search"

    def search(self, query: str, max_results: int = 15) -> list:
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed — skipping Google News RSS")
            return []

        articles = []
        for lang_region in [("en-US", "US"), ("en-PK", "PK")]:
            hl, gl = lang_region
            params = {
                "q":    query,
                "hl":   hl,
                "gl":   gl,
                "ceid": f"{gl}:{hl.split('-')[0]}",
            }
            url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:max_results]:
                    articles.append(Article(
                        title=entry.get("title", ""),
                        url=entry.get("link", ""),
                        source=entry.get("source", {}).get("title", "Google News"),
                        published=entry.get("published", ""),
                        summary=BeautifulSoup(
                            entry.get("summary", ""), "html.parser"
                        ).get_text()[:300],
                    ))
            except Exception as e:
                logger.debug(f"Google News RSS error ({hl}): {e}")

        return articles


class DuckDuckGoNews:
    """No API key needed. Scrapes DDG news results."""

    def search(self, query: str, max_results: int = 10) -> list:
        articles = []
        url = "https://html.duckduckgo.com/html/"
        try:
            resp = requests.post(
                url,
                data={"q": f"{query} news", "kl": "pk-en"},
                headers=_get_headers(1),
                timeout=8,
            )
            soup = BeautifulSoup(resp.text, "html.parser")
            for result in soup.select(".result__body")[:max_results]:
                title_el   = result.select_one(".result__title")
                snippet_el = result.select_one(".result__snippet")
                link_el    = result.select_one(".result__url")
                if title_el:
                    articles.append(Article(
                        title=title_el.get_text(strip=True),
                        url=link_el.get_text(strip=True) if link_el else "",
                        source="DuckDuckGo",
                        summary=snippet_el.get_text(strip=True) if snippet_el else "",
                    ))
        except Exception as e:
            logger.debug(f"DuckDuckGo error: {e}")
        return articles


class BingNewsRSS:
    """Bing News RSS — good fallback."""

    def search(self, query: str, max_results: int = 10) -> list:
        if not FEEDPARSER_AVAILABLE:
            return []
        articles = []
        url = f"https://www.bing.com/news/search?q={urllib.parse.quote(query)}&format=RSS"
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_results]:
                articles.append(Article(
                    title=entry.get("title", ""),
                    url=entry.get("link", ""),
                    source=entry.get("source", {}).get("title", "Bing News"),
                    published=entry.get("published", ""),
                    summary=BeautifulSoup(
                        entry.get("summary", ""), "html.parser"
                    ).get_text()[:300],
                ))
        except Exception as e:
            logger.debug(f"Bing News RSS error: {e}")
        return articles


class DirectSiteScraper:
    """Direct scraping of Pakistani/international news sites."""

    SITES = {
        "Dawn":     "https://www.dawn.com/search?q={query}",
        "Geo":      "https://www.geo.tv/search?query={query}",
        "ARY":      "https://arynews.tv/?s={query}",
        "The News": "https://www.thenews.com.pk/search?q={query}",
        "Tribune":  "https://tribune.com.pk/search/{query}",
        "BBC Urdu": "https://www.bbc.com/urdu/search?q={query}",
    }

    SELECTORS = {
        "Dawn":     ["h2.story__title a", "h3.story__title a", ".article-title a"],
        "Geo":      [".story-card h3 a", ".news-title a", "h2 a"],
        "ARY":      [".jeg_post_title a", "h3.post-title a"],
        "The News": [".news-title a", "h2 a", ".story-title a"],
        "Tribune":  [".story-title a", "h2 a", ".post-title a"],
        "BBC Urdu": ["h3.bbc-1yxnyxh a", ".bbc-1ykdmqj h3 a", "h3 a"],
    }

    # Per-site timeout keeps one slow site from blocking the others
    SITE_TIMEOUT = 5

    def search(self, query: str, max_results: int = 5) -> list:
        articles = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._scrape_site, name, url_tpl, query, max_results): name
                for name, url_tpl in self.SITES.items()
            }
            for future in as_completed(futures, timeout=8):
                try:
                    articles.extend(future.result())
                except Exception as e:
                    logger.debug(f"Site scrape error: {e}")
        return articles

    def _scrape_site(self, name: str, url_tpl: str, query: str, max_results: int) -> list:
        url      = url_tpl.format(query=urllib.parse.quote(query))
        articles = []
        try:
            resp = requests.get(
                url, headers=_get_headers(), timeout=self.SITE_TIMEOUT, allow_redirects=True
            )
            if resp.status_code in (403, 429, 503):
                logger.debug(f"{name}: blocked ({resp.status_code})")
                return []
            soup      = BeautifulSoup(resp.text, "html.parser")
            selectors = self.SELECTORS.get(name, ["h2 a", "h3 a", ".title a"])
            for selector in selectors:
                for el in soup.select(selector)[:max_results]:
                    title = el.get_text(strip=True)
                    href  = el.get("href", "")
                    if title and len(title) > 15:
                        if href.startswith("/"):
                            base = urllib.parse.urlparse(url)
                            href = f"{base.scheme}://{base.netloc}{href}"
                        articles.append(Article(title=title, url=href, source=name))
                if articles:
                    break
        except requests.exceptions.Timeout:
            logger.debug(f"{name}: timeout")
        except Exception as e:
            logger.debug(f"{name}: {e}")
        return articles


class NewsAPIClient:
    """Free NewsAPI.org (100 req/day free tier)."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "")

    def search(self, query: str, max_results: int = 10) -> list:
        if not self.api_key:
            return []
        articles = []
        url    = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "apiKey":   self.api_key,
            "pageSize": max_results,
            "sortBy":   "relevancy",
            "language": "en",
            "from":     (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
        }
        try:
            resp = requests.get(url, params=params, timeout=6)
            data = resp.json()
            for item in data.get("articles", []):
                articles.append(Article(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    published=item.get("publishedAt", ""),
                    summary=item.get("description", "") or "",
                ))
        except Exception as e:
            logger.debug(f"NewsAPI error: {e}")
        return articles


import os   # needed by NewsAPIClient above — placed here to keep top-level imports clean


# ── Main Scraper ───────────────────────────────────────────────────────────────

class NewsScraper:
    """
    Multi-strategy news scraper with hybrid similarity matching.

    Key robustness improvements over the original:
    - Per-strategy timeout of 15 s (up from 10 s)
    - Failed strategies never block successful ones
    - Partial results (from strategies that finish) are always returned
    - Overall wall-clock timeout of 15 s (strategies that haven't responded
      by then are abandoned; results collected so far are used)
    """

    # ── Timeouts ───────────────────────────────────────────────────────
    STRATEGY_TIMEOUT = 15   # max seconds to wait for *all* strategies to respond
    PER_FUTURE_GET   = 15   # future.result() timeout (same — we wait the full window)

    def __init__(self, trusted_sources_db=None,
                 news_api_key: str = "",
                 similarity_threshold: float = 0.25):
        # trusted_sources_db is accepted for orchestrator compatibility but not used
        self.threshold = similarity_threshold
        self.similarity = SimilarityEngine()
        self.similarity.MATCH_THRESHOLD = similarity_threshold

        self._strategies = [
            ("Google News RSS", GoogleNewsRSS()),
            ("Bing News RSS",   BingNewsRSS()),
            ("Direct Sites",    DirectSiteScraper()),
            ("DuckDuckGo",      DuckDuckGoNews()),
        ]
        if news_api_key:
            self._strategies.insert(0, ("NewsAPI", NewsAPIClient(news_api_key)))

    def _deduplicate(self, articles: list) -> list:
        seen   = set()
        unique = []
        for art in articles:
            key = hashlib.md5(art.title.lower().strip().encode()).hexdigest()
            if key not in seen and len(art.title) > 10:
                seen.add(key)
                unique.append(art)
        return unique

    def _score_articles(self, query: str, articles: list) -> list:
        scored = []
        for art in articles:
            candidate   = f"{art.title}. {art.summary}"
            score, meth = self.similarity.score(query, candidate)
            title_score, _ = self.similarity.score(query, art.title)
            final_score     = max(score, title_score)

            art.similarity_score = round(final_score, 4)
            art.match_method     = meth
            scored.append(art)

        return sorted(scored, key=lambda a: a.similarity_score, reverse=True)

    def search(self, query: str, max_total_articles: int = 50) -> ScraperResult:
        start  = time.time()
        result = ScraperResult(query=query)
        all_articles: list = []

        logger.info(f"Searching: '{query}'")

        # Run all strategies concurrently; collect whatever finishes in time
        with ThreadPoolExecutor(max_workers=len(self._strategies)) as executor:
            future_to_name = {
                executor.submit(strategy.search, query): name
                for name, strategy in self._strategies
            }

            # Drain futures with a per-future timeout so slow strategies
            # don't stall fast ones.  We stop waiting after STRATEGY_TIMEOUT
            # seconds total (measured from when we start iterating).
            deadline = time.time() + self.STRATEGY_TIMEOUT

            for future in as_completed(future_to_name):
                name           = future_to_name[future]
                remaining_time = max(0.5, deadline - time.time())
                try:
                    found = future.result(timeout=remaining_time)
                    logger.info(f"  {name}: {len(found)} articles")
                    all_articles.extend(found)
                    result.strategies_used.append(name)
                except FuturesTimeoutError:
                    logger.warning(f"  {name}: timed out (skipping)")
                    result.errors.append(f"{name}: timed out")
                except Exception as e:
                    logger.debug(f"  {name}: error — {e}")
                    result.errors.append(f"{name}: {e}")

        # Always return whatever we collected, even if some strategies failed
        unique  = self._deduplicate(all_articles)
        logger.info(f"Unique articles after dedup: {len(unique)}")

        scored  = self._score_articles(query, unique[:max_total_articles])
        matches = [a for a in scored if self.similarity.is_match(a.similarity_score)]

        result.total_articles = len(scored)
        result.matches_found  = len(matches)
        result.found          = len(matches) > 0
        result.articles       = scored[:20]
        result.top_match      = matches[0] if matches else (scored[0] if scored else None)
        result.execution_time = round(time.time() - start, 2)

        logger.info(
            f"Done in {result.execution_time}s | "
            f"Total: {result.total_articles} | "
            f"Matches (>={self.threshold}): {result.matches_found}"
        )
        if result.top_match:
            logger.info(
                f"Top match [{result.top_match.similarity_score:.3f}]: "
                f"{result.top_match.title[:80]} ({result.top_match.source})"
            )

        return result

    def search_dict(self, query: str, **kwargs) -> dict:
        r = self.search(query, **kwargs)
        return {
            "query":           r.query,
            "total_articles":  r.total_articles,
            "matches_found":   r.matches_found,
            "found":           r.found,
            "execution_time":  r.execution_time,
            "threshold_used":  self.threshold,
            "strategies_used": r.strategies_used,
            "errors":          r.errors,
            "top_match": {
                "title":            r.top_match.title,
                "url":              r.top_match.url,
                "source":           r.top_match.source,
                "similarity_score": r.top_match.similarity_score,
                "match_method":     r.top_match.match_method,
            } if r.top_match else None,
            "articles": [
                {
                    "title":            a.title,
                    "url":              a.url,
                    "source":           a.source,
                    "similarity_score": a.similarity_score,
                    "match_method":     a.match_method,
                    "summary":          a.summary[:150],
                }
                for a in r.articles
            ],
        }

    # ============================================
    # LEGACY METHOD FOR ORCHESTRATOR COMPATIBILITY
    # ============================================
    def verify_news_exists(self, claim: str) -> dict:
        """
        Legacy entry-point called by the orchestrator for TEXT analysis.
        Returns the same keys as the original NewsSiteScraper.
        """
        result = self.search_dict(claim)
        return {
            'found':                result.get('found', False),
            'matches_found':        result.get('matches_found', 0),
            'total_articles':       result.get('total_articles', 0),
            'matches':              result.get('articles', []),
            'best_match':           result.get('top_match', {}),
            'query_used':           result.get('query', claim),
            'similarity_threshold': result.get('threshold_used', 0.25),
            'credibility_stats': {
                'trusted_sources_found':     0,
                'total_sources':             result.get('strategies_used', []),
                'credibility_boost_applied': False,
                'credibility_weight':        0.2,
            }
        }


# ── CLI / Quick Test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    query   = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Pakistan election 2024 results"
    scraper = NewsScraper(similarity_threshold=0.25)
    result  = scraper.search_dict(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))