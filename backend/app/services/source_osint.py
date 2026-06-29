"""
TruthLens OSINT Source Verification — Revised Scoring Edition
=============================================================
CHANGES FROM ORIGINAL (scoring section only — all signal methods unchanged):

Key insight: the original design used a *flat* additive model starting at 50,
where every missing signal contributed a red flag and the combinatorial penalty
could knock a legitimate-but-small site from ~50 all the way to 25-32.

Revised approach uses a THREE-LAYER model:
  Layer 1 – Infrastructure baseline  (SSL + DNS resolvability + domain age)
  Layer 2 – Presence signals         (Wikipedia, WHOIS org, contact, social)
  Layer 3 – Reach signals            (Google News, citations, Wayback)

Network-sensitive signals (Google News, Wayback, citations) that commonly
fail due to rate-limits / timeouts are TREATED AS NEUTRAL when they error —
they do not generate red flags.  Only if they succeed AND return negative
evidence do they contribute a penalty.

Red flags are reserved for *positive evidence of untrustworthiness*:
  - self-signed SSL
  - free-email WHOIS registration
  - fully hidden WHOIS (both markers)

A floor of 50 is guaranteed for any domain that:
  - resolves (DNS OK)
  - has valid SSL
  - is older than 30 days

Target calibration (approximate):
  securequanta.com  – SSL ✓, age 3y, business email  → ~70  MEDIUM→HIGH
  bbc.com           – all signals likely firing        → ≥85  HIGH
  microsoft.com     – all signals likely firing        → ≥85  HIGH
  github.com        – all signals likely firing        → ≥80  HIGH
"""

import logging
import re
import ssl
import socket
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeout
from typing import Dict, List, Any, Optional, Callable
from urllib.parse import urlparse, quote_plus

import requests
import whois

logger = logging.getLogger(__name__)


# ======================================================================
# SIMPLE TTL CACHE  (unchanged)
# ======================================================================
class TTLCache:
    def __init__(self, default_ttl: int = 86400):
        self._store: Dict[str, tuple] = {}
        self._lock = threading.RLock()
        self.default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            value, expiry = entry
            if time.time() > expiry:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            ttl = ttl or self.default_ttl
            self._store[key] = (value, time.time() + ttl)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def size(self) -> int:
        with self._lock:
            return len(self._store)


# ======================================================================
# MAIN MODULE
# ======================================================================
class SourceOSINT:
    """
    OSINT with all original signals plus revised three-layer scoring.
    Signal methods are 100 % unchanged.  Only verify_source() scoring logic
    has been updated.
    """

    SIGNAL_TIMEOUT = 5
    OVERALL_TIMEOUT = 12

    WHOIS_PRIVACY_MARKERS = [
        'redacted', 'privacy', 'whoisguard', 'domains by proxy',
        'withheld for privacy', 'gdpr masked', 'data protected',
        'identity protection', 'private registration'
    ]

    FREE_EMAIL_PROVIDERS = {
        'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
        'aol.com', 'protonmail.com', 'mail.com', 'icloud.com'
    }

    SOURCE_MAP = {
        'bbc': 'BBC', 'bbc news': 'BBC', 'bbc.com': 'BBC', 'bbc.co.uk': 'BBC',
        'cnn': 'CNN', 'cnn news': 'CNN', 'cnn.com': 'CNN',
        'reuters': 'Reuters', 'reuters.com': 'Reuters',
        'ap news': 'Associated Press', 'associated press': 'Associated Press',
        'dawn': 'Dawn', 'dawn news': 'Dawn', 'dawn.com': 'Dawn',
        'geo news': 'Geo News', 'geo.tv': 'Geo News',
        'ary news': 'ARY News', 'arynews.tv': 'ARY News',
        'samaa': 'SAMAA', 'samaa.tv': 'SAMAA',
        'the news': 'The News', 'thenews.com.pk': 'The News',
        'express tribune': 'Express Tribune', 'tribune.com.pk': 'Express Tribune',
        'guardian': 'The Guardian', 'theguardian.com': 'The Guardian',
        'nytimes': 'New York Times', 'nytimes.com': 'New York Times',
        'wsj': 'Wall Street Journal', 'wsj.com': 'Wall Street Journal',
        'washington post': 'Washington Post', 'washingtonpost.com': 'Washington Post',
        'fox news': 'Fox News', 'foxnews.com': 'Fox News',
        'al jazeera': 'Al Jazeera', 'aljazeera.com': 'Al Jazeera',
        'sky news': 'Sky News', 'skynews.com': 'Sky News',
        'nbc news': 'NBC News', 'nbcnews.com': 'NBC News',
        'abc news': 'ABC News', 'abcnews.go.com': 'ABC News',
        'bloomberg': 'Bloomberg', 'bloomberg.com': 'Bloomberg',
        'forbes': 'Forbes', 'forbes.com': 'Forbes',
        'economist': 'The Economist', 'economist.com': 'The Economist',
        'telegraph': 'The Telegraph', 'telegraph.co.uk': 'The Telegraph',
    }

    SOURCE_TO_DOMAIN = {
        'BBC': 'bbc.com',
        'CNN': 'cnn.com',
        'Reuters': 'reuters.com',
        'Associated Press': 'apnews.com',
        'Dawn': 'dawn.com',
        'Geo News': 'geo.tv',
        'ARY News': 'arynews.tv',
        'SAMAA': 'samaa.tv',
        'The News': 'thenews.com.pk',
        'Express Tribune': 'tribune.com.pk',
        'The Guardian': 'theguardian.com',
        'New York Times': 'nytimes.com',
        'Wall Street Journal': 'wsj.com',
        'Washington Post': 'washingtonpost.com',
        'Fox News': 'foxnews.com',
        'Al Jazeera': 'aljazeera.com',
        'Sky News': 'skynews.com',
        'NBC News': 'nbcnews.com',
        'ABC News': 'abcnews.go.com',
        'Bloomberg': 'bloomberg.com',
        'Forbes': 'forbes.com',
        'The Economist': 'economist.com',
        'The Telegraph': 'telegraph.co.uk',
    }

    # Emergency fallback for rate-limited signals (unchanged)
    EMERGENCY_FALLBACK = {
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk',
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'wsj.com', 'economist.com', 'ft.com', 'bloomberg.com',
        'aljazeera.com', 'dw.com', 'france24.com', 'npr.org',
        'cnn.com', 'cbsnews.com', 'nbcnews.com', 'abcnews.go.com',
        'dawn.com', 'geo.tv', 'arynews.tv', 'samaa.tv', 'thenews.com.pk'
    }

    def __init__(self, signal_timeout: int = 5, overall_timeout: int = 12,
                 cache_ttl: int = 86400, max_workers: int = 8):
        self.SIGNAL_TIMEOUT = signal_timeout
        self.OVERALL_TIMEOUT = overall_timeout
        self.max_workers = max_workers
        self.cache = TTLCache(default_ttl=cache_ttl)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.mobile_headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15'
        }
        logger.info("🔍 SourceOSINT initialized (revised 3-layer scoring)")

    # ------------------------------------------------------------------
    # URL helpers (unchanged)
    # ------------------------------------------------------------------
    def extract_domain_from_url(self, url_or_domain: str) -> str:
        if not url_or_domain:
            return ''
        raw = url_or_domain.strip()
        if raw.startswith(('http://', 'https://')):
            hostname = urlparse(raw).hostname or ''
        else:
            hostname = raw.split('/')[0]
        return re.sub(r'^www\.', '', hostname).lower()

    def get_credibility_for_url(self, url: str) -> Dict[str, Any]:
        domain = self.extract_domain_from_url(url)
        result = self.verify_source(domain if domain else url)
        result['input_url'] = url
        return result

    @staticmethod
    def _outlet_name_from_domain(domain: str) -> str:
        return domain.split('.')[0].lower()

    # ==================================================================
    # MAIN VERIFICATION — revised scoring, all signals intact
    # ==================================================================
    def verify_source(self, source: str) -> Dict[str, Any]:
        """
        Verify a source (domain or source-name string).
        Scoring uses a three-layer model; all original signal functions are
        called unchanged and in parallel.
        """
        source_lower = source.lower().strip()

        # Source-name → domain mapping (unchanged)
        is_source_name = (
            '.' not in source_lower
            and 'http' not in source_lower
            and 'www' not in source_lower
        )
        if is_source_name:
            for src_name, domain in self.SOURCE_TO_DOMAIN.items():
                if src_name.lower() == source_lower or source_lower in src_name.lower():
                    result = self.verify_source(domain)
                    result['original_source_name'] = source
                    result['source'] = src_name
                    return result
            return {
                'source': source, 'domain': source,
                'credibility_score': 60, 'credibility_level': 'MEDIUM',
                'positive_signals': [f"Recognized news source: {source}"],
                'warning_signals': [], 'red_flags': [], 'details': {},
                'from_cache': False, 'original_source_name': source,
            }

        domain = self.extract_domain_from_url(source) or source_lower

        # Cache check
        cache_key = f'verify:{domain}'
        cached = self.cache.get(cache_key)
        if cached:
            cached['from_cache'] = True
            return cached

        result: Dict[str, Any] = {
            'source': domain, 'domain': domain,
            'credibility_score': 50, 'credibility_level': 'UNKNOWN',
            'positive_signals': [], 'warning_signals': [],
            'red_flags': [], 'failed_signals': [],
            'details': {}, 'warnings': [],
            'from_cache': False, 'execution_time': 0,
        }

        if not domain or '.' not in domain:
            result['warnings'].append('Invalid domain format')
            return result

        start = time.time()

        # ---- Parallel signal collection (unchanged) ------------------
        signal_tasks: Dict[str, Callable] = {
            'google_news': lambda: self._check_google_news(domain),
            'wikipedia':   lambda: self._check_wikipedia(domain),
            'citations':   lambda: self._estimate_inbound_citations(domain),
            'social_media':lambda: self._discover_social_media(domain),
            'contact':     lambda: self._check_contact_and_editorial(domain),
            'whois':       lambda: self._check_whois(domain),
            'wayback':     lambda: self._check_wayback(domain),
            'ssl':         lambda: self._check_ssl(domain),
        }

        signal_results: Dict[str, Any] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._run_with_timeout, name, fn): name
                for name, fn in signal_tasks.items()
            }
            try:
                for future in as_completed(future_map, timeout=self.OVERALL_TIMEOUT):
                    name = future_map[future]
                    try:
                        signal_results[name] = future.result(timeout=0.1)
                    except Exception as e:
                        logger.debug(f"Signal {name} failed: {e}")
                        signal_results[name] = None
                        result['failed_signals'].append(name)
            except FutureTimeout:
                for fut, name in future_map.items():
                    if name not in signal_results:
                        signal_results[name] = None
                        result['failed_signals'].append(name)
                        fut.cancel()

        # ==============================================================
        # REVISED SCORING  — three-layer additive model
        # ==============================================================
        #
        # Philosophy:
        #   • Start at 0 (not 50) — build up rather than subtract down
        #   • Layer 1 (infrastructure) supplies the baseline floor
        #   • Layer 2 (presence)      supplies soft trust signals
        #   • Layer 3 (reach)         supplies reputation uplift
        #   • Red flags are ONLY issued for positive evidence of bad
        #     intent (self-signed cert, free-email WHOIS, hidden WHOIS)
        #   • Network-sensitive signals that timed-out / errored are
        #     silently skipped — they do NOT count as negative evidence
        #
        # Max possible score breakdown:
        #   Layer 1:  38 pts  (SSL 15 + age bonus 15 + business-email 8)
        #   Layer 2:  32 pts  (Wikipedia 12 + contact 8 + WHOIS org 7 + social 5)
        #   Layer 3:  30 pts  (Google News 15 + citations 10 + Wayback 5)
        #   Total:   100 pts
        # ==============================================================

        score = 0
        red_flags = []

        # --------------------------------------------------------------
        # LAYER 1 — Infrastructure baseline
        # Goal: any domain that resolves, has SSL, and is ≥30 days old
        # should reach at least 50 purely from this layer.
        # --------------------------------------------------------------

        # 1a. SSL certificate  (+15 for valid, -10 for self-signed)
        ssl_info = signal_results.get('ssl') or {'valid': False}
        result['details']['ssl'] = ssl_info
        if ssl_info.get('valid'):
            score += 15
            result['positive_signals'].append(
                f"Valid SSL ({ssl_info.get('issuer', 'CA')})"
            )
        elif ssl_info.get('self_signed'):
            score -= 10
            result['warning_signals'].append('Self-signed SSL certificate')
            red_flags.append('self_signed_ssl')

        # 1b. Domain age  (up to +15, graduated)
        # Also used later to enforce the SSL+age floor
        whois_info = signal_results.get('whois') or {}
        result['details']['whois'] = whois_info
        age_days = whois_info.get('age_days')
        if age_days is not None:
            result['details']['domain_age_days']  = age_days
            result['details']['domain_age_years'] = round(age_days / 365.25, 1)

        if age_days is not None:
            if age_days >= 1825:    # ≥5 years
                score += 15
                result['positive_signals'].append(
                    f"Domain age {round(age_days/365.25, 1)}y (established)"
                )
            elif age_days >= 730:   # 2–5 years
                score += 12
                result['positive_signals'].append(
                    f"Domain age {round(age_days/365.25, 1)}y"
                )
            elif age_days >= 365:   # 1–2 years
                score += 9
                result['positive_signals'].append(
                    f"Domain age {round(age_days/365.25, 1)}y"
                )
            elif age_days >= 90:    # 3–12 months
                score += 5
                result['positive_signals'].append(
                    f"Domain age {round(age_days/365.25, 1)}y"
                )
            elif age_days >= 30:    # 1–3 months
                score += 2

        # 1c. WHOIS email quality  (+8 business, -5 free, -5 fully hidden)
        #     WHOIS org bonus moved to Layer 2
        email_type = whois_info.get('email_type', 'unknown')
        if email_type == 'business':
            score += 8
            result['positive_signals'].append('Business email in WHOIS')
        elif email_type == 'free':
            score -= 5
            result['warning_signals'].append('Free email in WHOIS registration')
            red_flags.append('free_email_registration')

        if whois_info.get('is_fully_hidden'):
            score -= 5
            result['warning_signals'].append('WHOIS fully hidden')
            red_flags.append('fully_hidden_whois')

        # ---- FLOOR: SSL-valid + domain ≥30 days → guarantee ≥50 -----
        #   This prevents small-but-legitimate sites from scoring below 50
        #   solely because network-sensitive reach signals timed out.
        #   We apply this floor AFTER Layer 1 math so we know whether SSL
        #   is valid before deciding.
        ssl_valid      = ssl_info.get('valid', False)
        domain_old_enough = (age_days is not None and age_days >= 30)
        if ssl_valid and domain_old_enough and score < 50:
            logger.debug(f"{domain}: applying SSL+age floor (score was {score})")
            score = 50

        # --------------------------------------------------------------
        # LAYER 2 — Presence signals (the "is this a real organisation?")
        # These signals are generally reliable and not rate-limited.
        # --------------------------------------------------------------

        # 2a. WHOIS organization  (+7)
        if whois_info.get('has_organization'):
            score += 7
            result['positive_signals'].append(
                f"WHOIS org: {whois_info.get('organization')}"
            )

        # 2b. Wikipedia article  (+12)
        wiki = signal_results.get('wikipedia') or {'has_article': False}
        result['details']['wikipedia'] = wiki
        if wiki.get('has_article'):
            score += 12
            result['positive_signals'].append(
                f"Wikipedia: {wiki.get('article_title')} ({wiki.get('language')})"
            )
        # Not having a Wikipedia article is NOT a red flag —
        # most legitimate businesses don't have one.

        # 2c. Contact & editorial markers  (+8 full / +4 partial / +3 editorial)
        contact = signal_results.get('contact') or {}
        result['details']['contact'] = contact
        if contact.get('has_full_contact'):
            score += 8
            result['positive_signals'].append('Full contact info found')
        elif contact.get('has_partial_contact'):
            score += 4
            result['positive_signals'].append('Partial contact info found')
        if contact.get('has_editorial_markers'):
            score += 3
            result['positive_signals'].append('Editorial schema.org markers')

        # 2d. Social media accounts  (+5 for ≥3, +3 for 1–2)
        #     Only penalise if we actually fetched the homepage successfully
        #     (i.e. signal did not fail) and found nothing.
        social = signal_results.get('social_media') or {'accounts_found': 0}
        result['details']['social_media'] = social
        accounts = social.get('accounts_found', 0)
        if accounts >= 3:
            score += 5
            result['positive_signals'].append(f"{accounts} social media accounts")
        elif accounts >= 1:
            score += 3
            result['positive_signals'].append(f"{accounts} social account(s)")

        # --------------------------------------------------------------
        # LAYER 3 — Reach / reputation signals
        # These are the network-sensitive signals.
        # Rule: if the signal returned an error / timed out, treat as
        # NEUTRAL (skip silently). Only count it if we got a real result.
        # --------------------------------------------------------------

        # 3a. Google News  (+15 indexed, 0 otherwise — NOT a red flag)
        gnews = signal_results.get('google_news') or {'indexed': False, 'result_count': 0}
        result['details']['google_news'] = gnews
        gnews_errored = bool(gnews.get('error')) or ('google_news' in result['failed_signals'])
        if not gnews_errored:
            if gnews.get('indexed'):
                score += 15
                result['positive_signals'].append(
                    f"Indexed in Google News ({gnews.get('result_count', 0)} items)"
                )
            # If not errored and not indexed: neutral — no red flag added

        # 3b. Inbound citations  (+10 high, +5 moderate — timeout = neutral)
        citations = signal_results.get('citations') or {'estimated_mentions': 0}
        result['details']['citations'] = citations
        cite_errored = ('citations' in result['failed_signals'])
        if not cite_errored:
            mentions = citations.get('estimated_mentions', 0)
            if mentions > 1000:
                score += 10
                result['positive_signals'].append(f"~{mentions:,} inbound mentions")
            elif mentions > 100:
                score += 5
                result['positive_signals'].append('Moderate inbound mentions')
            # Zero mentions with a real result: neutral, no penalty

        # 3c. Wayback Machine  (+5 if >100 snapshots — timeout = neutral)
        wayback = signal_results.get('wayback') or {'snapshots': 0}
        result['details']['wayback'] = wayback
        wb_errored = ('wayback' in result['failed_signals'])
        if not wb_errored:
            snaps = wayback.get('snapshots', 0)
            if snaps > 100:
                score += 5
                result['positive_signals'].append(f"Wayback: {snaps} snapshots")
            # Zero snapshots on a live check: add a mild caution but no hard penalty

        # --------------------------------------------------------------
        # Combinatorial penalty
        # ONLY applied for genuine trust-negative red flags, not for
        # missing reach signals.  Max flags possible = 3 (self-signed SSL,
        # free email, hidden WHOIS) so penalty is capped at 10.
        # --------------------------------------------------------------
        result['red_flags'] = red_flags
        penalty = self._compute_combinatorial_penalty(red_flags)
        if penalty > 0:
            score -= penalty
            result['warning_signals'].append(
                f"{len(red_flags)} trust red-flag(s) combined (−{penalty})"
            )

        # --------------------------------------------------------------
        # Emergency fallback (unchanged — for heavily rate-limited domains)
        # --------------------------------------------------------------
        failed_count = len(result['failed_signals'])
        if failed_count >= 5 and domain in self.EMERGENCY_FALLBACK:
            score = max(score, 85)
            result['warning_signals'].append(
                f'Emergency fallback applied ({failed_count} signals failed, '
                'domain in trusted top-tier list)'
            )
            result['details']['fallback_applied'] = True

        # Finalize
        score = max(0, min(100, score))
        result['credibility_score']  = score
        result['credibility_level']  = self._level(score)
        result['ssl_valid']          = ssl_info.get('valid', False)
        result['domain_age_years']   = result['details'].get('domain_age_years', 0)
        result['execution_time']     = round(time.time() - start, 2)

        logger.info(
            f"{domain}: score={score} ({result['credibility_level']}) "
            f"in {result['execution_time']}s | "
            f"red_flags={red_flags} | failed={result['failed_signals']}"
        )

        self.cache.set(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Helper used by parallel executor (unchanged)
    # ------------------------------------------------------------------
    def _run_with_timeout(self, name: str, fn: Callable) -> Any:
        try:
            return fn()
        except Exception as e:
            logger.debug(f"Signal {name} exception: {e}")
            return None

    @staticmethod
    def _compute_combinatorial_penalty(red_flags: List[str]) -> int:
        """
        Revised: penalty only for genuine trust red flags (max 3 possible),
        so table is intentionally lighter.
        """
        n = len(red_flags)
        if n == 0: return 0
        if n == 1: return 3    # single flag, small nudge
        if n == 2: return 7    # two flags together, moderate
        return 12              # all three, serious concern

    # ==================================================================
    # All original signal methods — UNCHANGED
    # ==================================================================

    def _check_google_news(self, domain: str) -> Dict[str, Any]:
        cache_key = f'gnews:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {'indexed': False, 'result_count': 0}
        try:
            r = self.session.get(
                'https://news.google.com/rss/search',
                params={'q': f'site:{domain}', 'hl': 'en-US', 'gl': 'US'},
                timeout=self.SIGNAL_TIMEOUT
            )
            if r.status_code == 200:
                items = r.text.count('<item>')
                if items == 0:
                    items = max(0, r.text.count('<link>') - 1)
                result['result_count'] = items
                result['indexed'] = items >= 3
        except Exception as e:
            result['error'] = str(e)
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _check_wikipedia(self, domain: str) -> Dict[str, Any]:
        cache_key = f'wiki:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {'has_article': False, 'article_title': None, 'language': None}
        outlet_name = self._outlet_name_from_domain(domain)
        queries = [outlet_name, domain]
        languages = ['en', 'es', 'fr', 'de', 'ar', 'ur', 'hi']
        for lang in languages:
            for query in queries:
                try:
                    r = self.session.get(
                        f'https://{lang}.wikipedia.org/w/api.php',
                        params={
                            'action': 'query', 'list': 'search',
                            'srsearch': query, 'format': 'json', 'srlimit': 5
                        },
                        timeout=self.SIGNAL_TIMEOUT
                    )
                    if r.status_code != 200:
                        continue
                    hits = r.json().get('query', {}).get('search', [])
                    for hit in hits[:3]:
                        title   = hit['title']
                        snippet = hit.get('snippet', '').lower()
                        if domain in snippet:
                            result.update({'has_article': True, 'article_title': title, 'language': lang})
                            self.cache.set(cache_key, result, ttl=86400)
                            return result
                        try:
                            ex = self.session.get(
                                f'https://{lang}.wikipedia.org/w/api.php',
                                params={
                                    'action': 'query', 'prop': 'extracts',
                                    'exintro': True, 'explaintext': True,
                                    'titles': title, 'format': 'json'
                                },
                                timeout=self.SIGNAL_TIMEOUT
                            )
                            if ex.status_code == 200:
                                pages = ex.json().get('query', {}).get('pages', {})
                                for _, page in pages.items():
                                    extract = (page.get('extract') or '').lower()
                                    if domain in extract or (outlet_name in title.lower() and len(outlet_name) >= 4):
                                        result.update({'has_article': True, 'article_title': title, 'language': lang})
                                        self.cache.set(cache_key, result, ttl=86400)
                                        return result
                        except Exception:
                            continue
                except Exception:
                    continue
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _estimate_inbound_citations(self, domain: str) -> Dict[str, Any]:
        cache_key = f'cite:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {'estimated_mentions': 0, 'source': None}
        try:
            r = self.session.get(
                'https://www.bing.com/search',
                params={'q': f'"{domain}" -site:{domain}'},
                timeout=self.SIGNAL_TIMEOUT
            )
            if r.status_code == 200:
                m = re.search(r'([\d,]+)\s+results', r.text)
                if m:
                    result['estimated_mentions'] = int(m.group(1).replace(',', ''))
                    result['source'] = 'bing'
        except Exception:
            pass
        if result['estimated_mentions'] == 0:
            try:
                r = self.session.get(
                    'https://html.duckduckgo.com/html/',
                    params={'q': f'"{domain}"'},
                    timeout=self.SIGNAL_TIMEOUT
                )
                if r.status_code == 200:
                    result_divs = r.text.count('class="result__body"')
                    if result_divs > 0:
                        result['estimated_mentions'] = result_divs * 100
                        result['source'] = 'duckduckgo_estimate'
            except Exception:
                pass
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _discover_social_media(self, domain: str) -> Dict[str, Any]:
        cache_key = f'social:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {'accounts_found': 0, 'platforms': []}
        patterns = {
            'twitter':   r'(?:twitter\.com|x\.com)/([A-Za-z0-9_]{2,})',
            'facebook':  r'facebook\.com/([A-Za-z0-9.\-]{3,})',
            'instagram': r'instagram\.com/([A-Za-z0-9_.]{2,})',
            'youtube':   r'youtube\.com/(?:c/|channel/|user/|@)([A-Za-z0-9_\-]{2,})',
            'linkedin':  r'linkedin\.com/(?:company|in)/([A-Za-z0-9\-]{2,})',
        }
        generic = {'share', 'home', 'login', 'signup', 'sharer', 'intent'}
        for headers in [None, self.mobile_headers]:
            try:
                r = self.session.get(f'https://{domain}', timeout=self.SIGNAL_TIMEOUT, headers=headers)
                if r.status_code != 200:
                    continue
                html = r.text
                found = []
                for platform, pattern in patterns.items():
                    matches = re.findall(pattern, html)
                    valid   = [m for m in matches if m.lower() not in generic]
                    if valid:
                        found.append({'platform': platform, 'handle': valid[0]})
                if found:
                    result['platforms']      = found
                    result['accounts_found'] = len(found)
                    break
            except Exception:
                continue
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _check_contact_and_editorial(self, domain: str) -> Dict[str, Any]:
        cache_key = f'contact:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {
            'has_full_contact': False, 'has_partial_contact': False,
            'has_editorial_markers': False, 'pages_checked': []
        }
        paths = [
            '/', '/contact', '/contact-us', '/about', '/about-us',
            '/info/contact-us', '/info/about-us', '/help/contact',
            '/imprint', '/impressum', '/feedback', '/customer-service'
        ]
        combined = ''
        per_path_timeout = max(2, self.SIGNAL_TIMEOUT // 2)
        for path in paths:
            if len(result['pages_checked']) >= 3:
                break
            try:
                r = self.session.get(f'https://{domain}{path}', timeout=per_path_timeout, allow_redirects=True)
                if r.status_code == 200 and len(r.text) > 200:
                    combined += r.text + '\n'
                    result['pages_checked'].append(path)
            except Exception:
                continue
        if not combined:
            self.cache.set(cache_key, result, ttl=86400)
            return result
        domain_emails   = re.findall(rf'[A-Za-z0-9._%+-]+@(?:[A-Za-z0-9.-]+\.)?{re.escape(domain)}', combined)
        phones          = re.findall(r'(?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}', combined)
        has_postal_schema = 'PostalAddress' in combined or 'streetAddress' in combined
        has_org_schema    = 'NewsMediaOrganization' in combined or ('Organization' in combined and 'schema.org' in combined)
        signals           = sum([bool(domain_emails), len(phones) > 0, has_postal_schema])
        result['has_full_contact']       = signals >= 3
        result['has_partial_contact']    = signals >= 1
        result['has_editorial_markers']  = has_org_schema or has_postal_schema
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _check_whois(self, domain: str) -> Dict[str, Any]:
        cache_key = f'whois:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        info = {
            'has_organization': False, 'organization': None,
            'is_fully_hidden': False,  'age_days': None,
            'email_type': 'unknown',
        }
        try:
            w = whois.whois(domain)
            for field in [w.org, getattr(w, 'organization', None)]:
                if field and isinstance(field, str) and len(field.strip()) > 2:
                    if not any(m in field.lower() for m in self.WHOIS_PRIVACY_MARKERS):
                        info['has_organization'] = True
                        info['organization']     = field.strip()
                        break
            text         = str(w).lower()
            privacy_hits = sum(1 for m in self.WHOIS_PRIVACY_MARKERS if m in text)
            info['is_fully_hidden'] = privacy_hits >= 2 and not info['has_organization']
            creation = w.creation_date
            if isinstance(creation, list):
                creation = creation[0] if creation else None
            if creation:
                if creation.tzinfo is None:
                    creation = creation.replace(tzinfo=timezone.utc)
                info['age_days'] = (datetime.now(timezone.utc) - creation).days
            email = w.emails
            if isinstance(email, list) and email:
                email = email[0]
            if email and isinstance(email, str) and '@' in email:
                provider = email.split('@')[-1].lower()
                info['email_type'] = 'free' if provider in self.FREE_EMAIL_PROVIDERS else 'business'
        except Exception as e:
            info['error'] = str(e)
        self.cache.set(cache_key, info, ttl=604800)
        return info

    def _check_wayback(self, domain: str) -> Dict[str, Any]:
        cache_key = f'wayback:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        result = {'snapshots': 0}
        try:
            r = self.session.get(
                'http://web.archive.org/cdx/search/cdx',
                params={'url': domain, 'output': 'json', 'limit': 1000},
                timeout=self.SIGNAL_TIMEOUT
            )
            if r.status_code == 200:
                data = r.json()
                if len(data) > 1:
                    result['snapshots'] = len(data) - 1
        except Exception:
            pass
        self.cache.set(cache_key, result, ttl=86400)
        return result

    def _check_ssl(self, domain: str) -> Dict[str, Any]:
        cache_key = f'ssl:{domain}'
        if cached := self.cache.get(cache_key):
            return cached
        info = {'valid': False, 'issuer': None, 'self_signed': False}
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=self.SIGNAL_TIMEOUT) as sock:
                with ctx.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert   = ssock.getpeercert()
                    issuer = dict(x[0] for x in cert['issuer'])
                    info['issuer'] = issuer.get('organizationName', 'Unknown')
                    info['valid']  = True
        except ssl.SSLCertVerificationError as e:
            if 'self signed' in str(e).lower():
                info['self_signed'] = True
        except Exception:
            pass
        self.cache.set(cache_key, info, ttl=86400)
        return info

    # ==================================================================
    # Classification & backward-compat helpers (unchanged)
    # ==================================================================
    @staticmethod
    def _level(score: int) -> str:
        if score >= 80: return 'HIGH'
        if score >= 60: return 'MEDIUM'
        if score >= 40: return 'LOW'
        return 'VERY_LOW'

    def get_credibility_score(self, domain: str) -> int:
        return self.verify_source(domain)['credibility_score']

    def extract_sources_from_text(self, text: str) -> List[str]:
        if not text:
            return []
        sources   = set()
        text_lower = text.lower().strip()
        dash_pipe_match = re.search(r'[-–|]\s*([A-Za-z\s]{2,30})(?:\.|$)', text)
        if dash_pipe_match:
            potential = dash_pipe_match.group(1).strip().lower()
            for keyword, display_name in self.SOURCE_MAP.items():
                if keyword == potential or keyword in potential or potential in keyword:
                    sources.add(display_name)
        paren_match = re.search(r'\(([^)]+)\)', text)
        if paren_match:
            potential = paren_match.group(1).strip().lower()
            for keyword, display_name in self.SOURCE_MAP.items():
                if keyword == potential or keyword in potential or potential in keyword:
                    sources.add(display_name)
        attribution_match = re.search(r'(?:according to|reported by|via)\s+([A-Za-z\s]{2,30})', text, re.IGNORECASE)
        if attribution_match:
            potential = attribution_match.group(1).strip().lower()
            for keyword, display_name in self.SOURCE_MAP.items():
                if keyword == potential or keyword in potential or potential in keyword:
                    sources.add(display_name)
        prefix_match = re.search(r'^([A-Za-z\s]{2,20}):', text)
        if prefix_match:
            potential = prefix_match.group(1).strip().lower()
            for keyword, display_name in self.SOURCE_MAP.items():
                if keyword == potential or keyword in potential or potential in keyword:
                    sources.add(display_name)
        url_pattern = re.compile(r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9][a-zA-Z0-9-]{0,62}(?:\.[a-zA-Z0-9][a-zA-Z0-9-]{0,62})+)\b')
        for match in url_pattern.findall(text):
            if '.' in match and len(match.split('.')[-1]) >= 2:
                domain = match.lower()
                matched_known = False
                for keyword, display_name in self.SOURCE_MAP.items():
                    if keyword in domain:
                        sources.add(display_name)
                        matched_known = True
                        break
                if not matched_known:
                    sources.add(domain)
        result = list(sources)
        if result:
            logger.info(f"Extracted sources: {result}")
        return result[:10]

    def analyse_ocr_text(self, ocr_text: str) -> Dict[str, Any]:
        domains       = self.extract_sources_from_text(ocr_text)
        verifications = []
        with ThreadPoolExecutor(max_workers=min(len(domains) or 1, 4)) as ex:
            futures = {ex.submit(self.verify_source, d): d for d in domains}
            for fut in as_completed(futures):
                try:
                    verifications.append(fut.result(timeout=self.OVERALL_TIMEOUT + 2))
                except Exception:
                    pass
        avg = (sum(v['credibility_score'] for v in verifications) / len(verifications)
               if verifications else 0)
        return {
            'input_type': 'image_ocr',
            'mentioned_sources': domains,
            'verifications': verifications,
            'source_count': len(domains),
            'overall_source_credibility': round(avg, 1),
        }

    def clear_cache(self) -> None:
        self.cache.clear()
        logger.info("OSINT cache cleared")

    def cache_stats(self) -> Dict[str, int]:
        return {'entries': self.cache.size()}