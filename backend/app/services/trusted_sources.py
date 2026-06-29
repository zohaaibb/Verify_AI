# app/services/trusted_sources.py
"""
Trusted Sources Module - Google News + Wikidata
Verifies claims against trusted news sources
"""

import hashlib
import logging
import requests
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from functools import lru_cache
from urllib.parse import quote, urlparse
from bs4 import BeautifulSoup
import json

from app.services.news_scraper import SimilarityEngine

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.35
SIMILARITY_THRESHOLD_FALLBACK = 0.25
RSS_CACHE_TTL = 300  # 5 minutes


def _has_keyword_overlap(claim: str, title: str, min_overlap: int = 2) -> bool:
    STOPWORDS = {
        'the','a','an','is','are','was','were','be','been','have','has',
        'had','do','does','did','will','would','could','should','may',
        'might','this','that','and','or','but','in','on','at','to',
        'for','of','with','by','from','it','its','he','she','they',
        'we','you','i','new','says','said','after','over','into',
        'also','just','more','than','when','what','where','which',
        'declares','declared','declare','announce','announced','announces',
        'global','world','international','national','local','official','officials',
        'emergency','crisis','situation','issue','concern',
        'health','public','government','minister','ministry','authority','authorities',
        'according','sources','source','statement','confirmed','confirms',
        'claims','alleged','reportedly','report','reports','reported',
        'people','country','countries','years','year','time','times',
        'first','second','third','last','next','back','long','high',
        'president','prime','secretary','general','director',
        'million','billion','thousand','percent',
    }
    claim_words = {
        w.lower() for w in re.findall(r'\b\w{4,}\b', claim)
        if w.lower() not in STOPWORDS
    }
    title_words = {
        w.lower() for w in re.findall(r'\b\w{4,}\b', title)
        if w.lower() not in STOPWORDS
    }
    if not claim_words:
        raw_c = {w.lower() for w in re.findall(r'\b\w{3,}\b', claim)}
        raw_t = {w.lower() for w in re.findall(r'\b\w{3,}\b', title)}
        return len(raw_c & raw_t) >= min_overlap
    overlap = claim_words & title_words
    return len(overlap) >= 1

# ======================================================================
# Tiered Domain Credibility System
# ======================================================================

DOMAIN_TIERS = {
    1: {
        'reuters.com', 'apnews.com', 'afp.com', 'bloomberg.com',
    },
    2: {
        'bbc.com', 'bbc.co.uk', 'theguardian.com', 'nytimes.com',
        'wsj.com', 'washingtonpost.com', 'economist.com', 'ft.com',
        'aljazeera.com', 'dw.com', 'france24.com',
        'who.int', 'un.org', 'cdc.gov', 'fda.gov', 'nih.gov',
        'gov.pk', 'europa.eu', 'nato.int', 'voanews.com', 'rferl.org',
    },
    3: {
        'dawn.com', 'thenews.com.pk', 'tribune.com.pk', 'brecorder.com',
        'thehindu.com', 'indianexpress.com', 'hindustantimes.com',
        'timesofindia.indiatimes.com', 'abcnews.go.com', 'nbcnews.com',
        'cbsnews.com', 'skynews.com', 'cnn.com', 'independent.co.uk',
        'telegraph.co.uk',
        'npr.org', 'pbs.org', 'propublica.org',
    },
    4: {
        'geo.tv', 'arynews.tv', 'samaa.tv', 'dunyanews.tv',
        'pakistantoday.com.pk', 'foxnews.com', 'dailymail.co.uk',
        'nypost.com', 'express.co.uk',
    },
}

TIER_SCORES = {1: 98, 2: 90, 3: 78, 4: 58, 5: 35}

_DOMAIN_TO_TIER: dict = {}
for tier, domains in DOMAIN_TIERS.items():
    for d in domains:
        _DOMAIN_TO_TIER[d] = tier


@lru_cache(maxsize=512)
def resolve_domain(url: str) -> dict:
    raw_domain = _extract_raw_domain(url)
    if not raw_domain:
        return _unknown_domain(url)

    if raw_domain in _DOMAIN_TO_TIER:
        tier = _DOMAIN_TO_TIER[raw_domain]
        return _make_domain_result(raw_domain, tier)

    parts = raw_domain.split('.')
    for i in range(1, len(parts) - 1):
        parent = '.'.join(parts[i:])
        if parent in _DOMAIN_TO_TIER:
            tier = _DOMAIN_TO_TIER[parent]
            return _make_domain_result(raw_domain, tier, matched_as=parent)

    tld = '.' + parts[-1] if parts else ''
    baseline_tier = 5
    if tld in ('.gov', '.mil'):
        baseline_tier = 2
    elif tld == '.edu':
        baseline_tier = 3

    return _unknown_domain(raw_domain, baseline_tier=baseline_tier)


def _extract_raw_domain(url: str) -> str:
    try:
        if 'google.com' in url and 'url?q=' in url:
            m = re.search(r'url\?q=([^&]+)', url)
            if m:
                url = m.group(1)
        parsed = urlparse(url if '://' in url else 'https://' + url)
        domain = parsed.netloc.lower()
        domain = re.sub(r'^www\.', '', domain)
        domain = domain.split(':')[0]
        return domain
    except Exception:
        return ''


def _make_domain_result(domain: str, tier: int, matched_as: str = None) -> dict:
    score = TIER_SCORES[tier]
    return {
        'domain': domain,
        'matched_as': matched_as or domain,
        'tier': tier,
        'credibility_score': score,
        'is_trusted': tier <= 3,
        'tier_label': _tier_label(tier),
    }


def _unknown_domain(domain: str, baseline_tier: int = 5) -> dict:
    score = TIER_SCORES[baseline_tier]
    tld = domain.rsplit('.', 1)[-1] if '.' in domain else ''
    force_trusted = tld in ('gov', 'mil')
    return {
        'domain': domain,
        'matched_as': None,
        'tier': baseline_tier,
        'credibility_score': score,
        'is_trusted': force_trusted or (baseline_tier <= 3),
        'tier_label': _tier_label(baseline_tier),
    }


def _tier_label(tier: int) -> str:
    return {
        1: 'Wire Service',
        2: 'Major International',
        3: 'Regional/National',
        4: 'Low Editorial Standards',
        5: 'Unknown/Blog',
    }.get(tier, 'Unknown')


# ======================================================================
# RSS-level cache
# ======================================================================

_rss_cache: dict = {}


class TrustedSourcesDB:
    """
    Trusted Sources Database - Searches Google News and other trusted sources
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        self.similarity = SimilarityEngine()
        self._nlp = None
        
        # Source name mapping for display + reverse lookup for domain cred
        self.source_names = {
            'bbc.com': 'BBC', 'bbc.co.uk': 'BBC', 'reuters.com': 'Reuters',
            'apnews.com': 'Associated Press', 'cnn.com': 'CNN',
            'theguardian.com': 'Guardian', 'nytimes.com': 'New York Times',
            'wsj.com': 'Wall Street Journal', 'dawn.com': 'Dawn',
            'geo.tv': 'Geo News', 'arynews.tv': 'ARY News', 'samaa.tv': 'SAMAA',
            'thenews.com.pk': 'The News', 'tribune.com.pk': 'Express Tribune',
            'brecorder.com': 'Business Recorder',
            'aljazeera.com': 'Al Jazeera', 'dw.com': 'Deutsche Welle',
            'nbcnews.com': 'NBC News', 'npr.org': 'NPR',
            'abcnews.go.com': 'ABC News', 'cbsnews.com': 'CBS News',
            'voanews.com': 'Voice of America', 'rferl.org': 'Radio Free Europe',
            'independent.co.uk': 'Independent',
            'thehindu.com': 'Hindu',
        }
        self._source_to_domain = self._build_source_to_domain()
        self._debunk_embeddings = None
        self._support_embeddings = None
        
        logger.info(f"📰 TrustedSourcesDB initialized with tiered domain system ({sum(len(d) for d in DOMAIN_TIERS.values())} domains)")
    
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_nlp(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = False
        return self._nlp if self._nlp is not False else None

    @staticmethod
    def _clean_html(raw: str) -> str:
        if not raw:
            return ""
        text = BeautifulSoup(raw, "html.parser").get_text(separator=" ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _normalize_source_name(name: str) -> str:
        return re.sub(r'^the\s+', '', name.strip().lower())

    def _build_source_to_domain(self) -> dict:
        reverse = {}
        for domain, name in self.source_names.items():
            normalized = self._normalize_source_name(name)
            reverse[normalized] = domain

        aliases = {
            'ap news': 'apnews.com',
            'associated press': 'apnews.com',
            'ap': 'apnews.com',
            'bbc news': 'bbc.com',
            'bbc uk': 'bbc.co.uk',
            'al jazeera english': 'aljazeera.com',
            'al jazeera': 'aljazeera.com',
            'deutsche welle': 'dw.com',
            'afp': 'afp.com',
            'agence france-presse': 'afp.com',
            'guardian': 'theguardian.com',
            'nbc news': 'nbcnews.com',
            'abc news': 'abcnews.go.com',
            'cbs news': 'cbsnews.com',
            'sky news': 'skynews.com',
            'new york times': 'nytimes.com',
            'wall street journal': 'wsj.com',
            'washington post': 'washingtonpost.com',
            'express tribune': 'tribune.com.pk',
            'the news': 'thenews.com.pk',
            'the news international': 'thenews.com.pk',
            'geo news': 'geo.tv',
            'ary news': 'arynews.tv',
            'dawn': 'dawn.com',
            'voice of america': 'voanews.com',
            'voa news': 'voanews.com',
            'radio free europe': 'rferl.org',
            'stat news': 'statnews.com',
            'npr': 'npr.org',
        }

        reverse.update(aliases)
        return reverse

    @staticmethod
    def _build_candidate_text(item: dict) -> str:
        title = item.get("title", "") or ""
        desc = item.get("description", "") or ""
        desc_clean = BeautifulSoup(desc, "html.parser").get_text(separator=" ")
        desc_clean = re.sub(r"\s+", " ", desc_clean).strip()
        return f"{title} {title} {desc_clean}"

    @staticmethod
    def _cached_search(query: str, search_fn) -> list:
        key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        now = time.time()
        if key in _rss_cache:
            result, ts = _rss_cache[key]
            if now - ts < RSS_CACHE_TTL:
                logger.debug(f"RSS cache hit for query: {query[:60]}")
                return result
        result = search_fn(query)
        _rss_cache[key] = (result, now)
        return result

    # ------------------------------------------------------------------
    # Query expansion
    # ------------------------------------------------------------------

    def _generate_query_variants(self, claim: str) -> list:
        queries = []

        entity_query = self._expand_query(claim, max_terms=6)
        queries.append(entity_query)

        action_query = self._extract_action_query(claim)
        if action_query and action_query.lower() != entity_query.lower():
            queries.append(action_query)

        short_claim = claim[:80].rsplit(' ', 1)[0]
        if short_claim not in queries:
            queries.append(short_claim)

        seen = set()
        unique = []
        for q in queries:
            qn = q.lower().strip()
            if qn not in seen and len(qn) > 5:
                seen.add(qn)
                unique.append(q)

        return unique[:3]

    def _expand_query(self, claim: str, max_terms: int = 6) -> str:
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(claim[:300])
            cached_entities = [ent.text for ent in doc.ents]
            cached_nouns = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
            terms = []
            for t in cached_entities + cached_nouns:
                if t.lower() not in [x.lower() for x in terms]:
                    terms.append(t)
            if terms:
                return ' '.join(terms[:max_terms])
        words = claim.split()
        return ' '.join(words[:max_terms])

    def _extract_action_query(self, claim: str) -> str:
        nlp = self._get_nlp()
        if nlp:
            doc = nlp(claim[:300])
            subject, verb, obj = '', '', ''
            for token in doc:
                if token.dep_ in ('nsubj', 'nsubjpass') and not subject:
                    subject = token.text
                if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'relcl') and not verb:
                    verb = token.lemma_
                if token.dep_ in ('dobj', 'attr', 'pobj') and not obj:
                    obj = token.text
            parts = [p for p in [subject, verb, obj] if p]
            if len(parts) >= 2:
                return ' '.join(parts)
        words = claim.split()
        return ' '.join(words[:6]) if len(words) > 6 else claim

    @staticmethod
    def _deduplicate_results(all_results: list) -> list:
        seen_urls = {}
        for item in all_results:
            url = item.get('link', '')
            norm_url = url.split('?')[0].split('#')[0].lower()
            if norm_url not in seen_urls:
                seen_urls[norm_url] = item
            else:
                existing = seen_urls[norm_url]
                if item.get('credibility_score', 0) > existing.get('credibility_score', 0):
                    seen_urls[norm_url] = item
        return list(seen_urls.values())

    def _collect_all_results(self, claim: str) -> list:
        queries = self._generate_query_variants(claim)
        all_results = []
        for q in queries:
            results = self._cached_search(q, self._search_google_news)
            all_results.extend(results)
            time.sleep(0.3)
        return self._deduplicate_results(all_results)

    # ------------------------------------------------------------------
    # Stance detection — two-pass: regex lexicon + embedding anchors
    # ------------------------------------------------------------------

    DEBUNK_ANCHORS = [
        "this claim is false and misleading",
        "experts say this is misinformation",
        "fact check: false",
        "there is no evidence for this claim",
        "scientists debunk this conspiracy theory",
        "the claim has been debunked",
    ]

    SUPPORT_ANCHORS = [
        "officials confirmed the announcement",
        "the government has officially announced",
        "according to official sources this is confirmed",
        "the report has been verified by authorities",
    ]

    def _get_stance_embeddings(self):
        if self._debunk_embeddings is None:
            try:
                model = getattr(self.similarity, '_model', None)
                if model:
                    self._debunk_embeddings = model.encode(
                        self.DEBUNK_ANCHORS, convert_to_numpy=True
                    )
                    self._support_embeddings = model.encode(
                        self.SUPPORT_ANCHORS, convert_to_numpy=True
                    )
            except Exception:
                self._debunk_embeddings = False
        return self._debunk_embeddings, self._support_embeddings

    def _detect_stance(self, claim: str, title: str, description: str) -> dict:
        title_lower = title.lower()
        desc_lower = description.lower()

        REFUTATION_MARKERS = [
            'false', 'fake', 'misleading', 'misinformation', 'disinformation',
            'debunked', 'fact check', 'fact-check', 'no evidence', 'not true',
            'incorrect', 'hoax', 'rumor', 'unverified', 'disputed',
            'without evidence', 'no proof', 'baseless', 'unfounded',
            'contrary to', 'inaccurate', 'correction', 'retraction',
            'never happened', 'did not happen', 'denies', 'denied',
            'conspiracy', 'debunking', 'myth', 'pseudoscience',
            'scientists say', 'experts say', 'health officials say',
        ]

        SUPPORT_MARKERS = [
            'confirmed', 'confirms', 'verified', 'verifies', 'official',
            'announced', 'announces', 'according to', 'reports that',
            'statement', 'spokesperson confirmed', 'government says',
            'ministry confirms', 'declared', 'declaration',
        ]

        refute_score = 0.0
        support_score = 0.0

        for marker in REFUTATION_MARKERS:
            if marker in title_lower:
                refute_score += 3.0
            elif marker in desc_lower:
                refute_score += 1.0

        for marker in SUPPORT_MARKERS:
            if marker in title_lower:
                support_score += 3.0
            elif marker in desc_lower:
                support_score += 1.0

        quoted_in_title = bool(re.search(
            r'["\u2018\u2019\u201c\u201d].{5,80}["\u2018\u2019\u201c\u201d]',
            title_lower
        ))
        if quoted_in_title:
            refute_score += 2.0

        pass1_label = None
        if refute_score > support_score * 1.2:
            pass1_label = 'REFUTE'
        elif support_score > refute_score * 1.2:
            pass1_label = 'SUPPORT'

        pass2_label = None
        try:
            debunk_embs, support_embs = self._get_stance_embeddings()
            model = getattr(self.similarity, '_model', None)
            if model and debunk_embs is not False and debunk_embs is not None:
                import numpy as np
                article_text = f"{title_lower} {title_lower} {desc_lower}"[:400]
                article_emb = model.encode([article_text], convert_to_numpy=True)[0]
                debunk_sims = [
                    float(np.dot(article_emb, d_emb) / (np.linalg.norm(article_emb) * np.linalg.norm(d_emb)))
                    for d_emb in debunk_embs
                ]
                support_sims = [
                    float(np.dot(article_emb, s_emb) / (np.linalg.norm(article_emb) * np.linalg.norm(s_emb)))
                    for s_emb in support_embs
                ]
                max_debunk = max(debunk_sims) if debunk_sims else 0
                max_support = max(support_sims) if support_sims else 0
                if max_debunk > max_support + 0.08:
                    pass2_label = 'REFUTE'
                    refute_score += 3.0
                elif max_support > max_debunk + 0.08:
                    pass2_label = 'SUPPORT'
                    support_score += 3.0
        except Exception:
            pass

        total = refute_score + support_score
        if total == 0:
            label = 'NEUTRAL'
            confidence = 0.0
        elif refute_score > support_score * 1.2:
            label = 'REFUTE'
            confidence = min(refute_score / max(total, 1), 1.0)
        elif support_score > refute_score * 1.2:
            label = 'SUPPORT'
            confidence = min(support_score / max(total, 1), 1.0)
        else:
            label = 'MIXED'
            confidence = 0.4

        return {
            'stance': label,
            'confidence': round(confidence, 3),
            'refute_score': round(refute_score, 2),
            'support_score': round(support_score, 2),
            'pass1_label': pass1_label,
            'pass2_label': pass2_label,
            'quoted_in_title': quoted_in_title,
        }

    # ------------------------------------------------------------------
    # Corroboration
    # ------------------------------------------------------------------

    TIER1_ORGS = {'Reuters', 'AP', 'AFP', 'Bloomberg'}

    @staticmethod
    def _compute_corroboration(matches: list) -> dict:
        trusted_orgs = set()
        tier1_trusted = set()
        for m in matches:
            if m.get('is_trusted'):
                org = m.get('source', '')
                trusted_orgs.add(org)
                if org in TrustedSourcesDB.TIER1_ORGS:
                    tier1_trusted.add(org)
        return {
            'trusted_org_count': len(trusted_orgs),
            'trusted_orgs': list(trusted_orgs),
            'tier1_count': len(tier1_trusted),
            'total_sources': len(set(m.get('source', '') for m in matches)),
        }

    # ------------------------------------------------------------------
    # Evidence verdict
    # ------------------------------------------------------------------

    def _compute_evidence_verdict(self, matches: list, claim: str) -> dict:
        if not matches:
            return {
                'final_score': 0,
                'coverage_score': 0,
                'alignment_score': 0,
                'stance_modifier': 0,
                'verdict': 'UNVERIFIED',
                'verdict_reason': 'No relevant articles found',
                'corroboration': {},
                'support_count': 0,
                'refute_count': 0,
            }

        corr = self._compute_corroboration(matches)

        n_trusted = corr['trusted_org_count']
        tier1_count = corr['tier1_count']
        coverage_score = min((n_trusted * 15) + (tier1_count * 10), 50)

        trusted_matches = [m for m in matches if m.get('is_trusted')]
        all_for_align = trusted_matches if trusted_matches else matches
        sims = [m.get('claim_similarity', 0) for m in all_for_align]
        avg_sim = sum(sims) / len(sims) if sims else 0
        max_sim = max(sims) if sims else 0
        alignment_score = min(int((avg_sim * 0.4 + max_sim * 0.6) * 50), 50)

        support_count = sum(1 for m in matches if m.get('stance') == 'SUPPORT')
        refute_count = sum(1 for m in matches if m.get('stance') == 'REFUTE')
        total_stanced = support_count + refute_count

        if total_stanced == 0:
            stance_modifier = 0
        else:
            stance_ratio = (support_count - refute_count) / total_stanced
            stance_modifier = int(stance_ratio * 25)

        raw_score = coverage_score + alignment_score + stance_modifier
        final_score = max(0, min(raw_score, 100))

        unique_trusted_articles = len([m for m in matches if m.get('is_trusted')])
        max_sim_trusted = max(
            (m.get('claim_similarity', 0) for m in matches if m.get('is_trusted')),
            default=0
        )

        if refute_count > support_count and refute_count >= 2:
            verdict = 'LIKELY_FALSE'
            reason = f"Refuted by {refute_count} sources across {n_trusted} trusted orgs"
        elif (final_score >= 75
              and n_trusted >= 2
              and unique_trusted_articles >= 2
              and max_sim_trusted >= 0.40):
            verdict = 'VERIFIED'
            reason = f"Confirmed by {n_trusted} independent trusted sources"
        elif (final_score >= 45
              and n_trusted >= 1
              and max_sim_trusted >= 0.30):
            verdict = 'PARTIALLY_VERIFIED'
            reason = f"Supported by {n_trusted} trusted source(s) with moderate alignment"
        elif final_score >= 25:
            verdict = 'UNCONFIRMED'
            reason = "Related coverage found but insufficient trusted corroboration"
        else:
            verdict = 'UNVERIFIED'
            reason = "No reliable corroborating coverage found"

        return {
            'final_score': final_score,
            'coverage_score': coverage_score,
            'alignment_score': alignment_score,
            'stance_modifier': stance_modifier,
            'verdict': verdict,
            'verdict_reason': reason,
            'corroboration': corr,
            'support_count': support_count,
            'refute_count': refute_count,
            'unique_trusted_articles': unique_trusted_articles,
            'max_sim_trusted': round(max_sim_trusted, 4),
        }

    # ------------------------------------------------------------------

    def check_claim_against_trusted(self, claim: str) -> Dict[str, Any]:
        """
        Check a claim against trusted sources using Google News
        
        Returns:
            Dict with verification status, matches, and confidence
        """
        start_time = time.time()
        
        result = {
            'success': True,
            'claim': claim,
            'verified': False,
            'match_score': 0,
            'match_quality': 'UNVERIFIED',
            'matches': [],
            'source_stats': {
                'total_sources': 0,
                'trusted_count': 0,
                'unique_sources': [],
                'trusted_sources': [],
                'source_counts': {}
            },
            'link_stats': {
                'total': 0,
                'search': 0,
                'direct': 0
            },
            'verification_note': '',
            'execution_stats': {
                'total_time': 0
            }
        }
        
        try:
            search_results = self._collect_all_results(claim)
            
            if search_results:
                matches = self._process_search_results(search_results, claim)
                result['matches'] = matches
                result['link_stats']['total'] = len(matches)
                
                # Analyze matches — new evidence verdict
                if matches:
                    verdict = self._compute_evidence_verdict(matches, claim)
                    result['match_score'] = verdict['final_score']
                    result['match_quality'] = self._get_match_quality(verdict['final_score'])
                    result['source_stats'] = self._aggregate_source_stats(matches)
                    result['verified'] = verdict['verdict'] in ('VERIFIED', 'PARTIALLY_VERIFIED')
                    result['verdict'] = verdict['verdict']
                    result['verdict_reason'] = verdict['verdict_reason']
                    result['evidence_breakdown'] = verdict
                    result['verification_note'] = verdict['verdict_reason']
                else:
                    result['verification_note'] = "No matching articles found"
            else:
                result['verification_note'] = "No search results found"
            
        except Exception as e:
            logger.error(f"Trusted sources check failed: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        result['execution_stats']['total_time'] = round(time.time() - start_time, 2)
        return result
    
    def _search_google_news(self, query: str) -> List[Dict]:
        """Search Google News RSS for the query"""
        results = []
        
        try:
            search_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
            response = self.session.get(search_url, timeout=10)
            
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                for item in root.findall('.//item'):
                    try:
                        title = item.find('title').text if item.find('title') is not None else ''
                        link = item.find('link').text if item.find('link') is not None else ''
                        description = item.find('description').text if item.find('description') is not None else ''
                        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
                        
                        source = self._extract_source_from_description(description)
                        domain_info = resolve_domain(link)
                        
                        # Fallback: if Google News returned a redirect URL (news.google.com),
                        # use the source name to look up the domain
                        if domain_info['tier'] == 5:
                            normalized = self._normalize_source_name(source)
                            fallback_domain = self._source_to_domain.get(normalized)
                            if fallback_domain:
                                domain_info = resolve_domain(fallback_domain)
                            elif source != 'Unknown':
                                logger.debug(
                                    f"Unresolved source: '{source}' "
                                    f"normalized='{normalized}' raw_domain='{domain_info['domain']}'"
                                )
                        
                        results.append({
                            'title': title[:200] if title else '',
                            'link': link,
                            'description': description[:500] if description else '',
                            'published': pub_date[:20] if pub_date else '',
                            'source': source,
                            'domain': domain_info['domain'],
                            'is_trusted': domain_info['is_trusted'],
                            'credibility_score': domain_info['credibility_score'],
                            'domain_tier': domain_info['tier'],
                            'link_type': 'search'
                        })
                    except Exception as e:
                        logger.debug(f"Error parsing item: {e}")
                        continue
                
                results = results[:15]
                
        except Exception as e:
            logger.error(f"Google News search failed: {e}")
        
        return results
    
    def _extract_source_from_description(self, description: str) -> str:
        """Extract source name from Google News description"""
        try:
            # Extract text between <font> tags
            font_match = re.search(r'<font[^>]*>([^<]+)</font>', description)
            if font_match:
                source = font_match.group(1)
                return source.strip()
            
            # Alternative: extract from the end of description
            parts = description.split('&nbsp;')
            if len(parts) > 1:
                source = parts[-1].replace('</font>', '').replace('<font', '')
                return source.strip()
            
        except Exception:
            pass
        
        return 'Unknown'
    
    def _extract_domain_from_url(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            domain = domain.replace('www.', '')
            
            # Handle google redirect URLs
            if 'google.com' in domain:
                # Try to extract actual domain from the URL
                if 'url?q=' in url:
                    actual = re.search(r'url\?q=([^&]+)', url)
                    if actual:
                        return self._extract_domain_from_url(actual.group(1))
            
            return domain.lower()
            
        except Exception:
            return ''
    
    def _process_search_results(self, results: List[Dict], claim: str) -> List[Dict]:
        """Process and enrich search results — keyword gate + adaptive threshold"""
        processed = []
        
        for item in results:
            source = item.get('source', 'Unknown')
            if source == 'Unknown' and item.get('domain'):
                source = self.source_names.get(item['domain'], item['domain'].split('.')[0].title())
            
            # Keyword gate ALWAYS required — blocks structural mimics
            # ("WHO declares X global emergency" matching wrong X)
            if not _has_keyword_overlap(claim, item.get('title', ''), min_overlap=2):
                continue
            
            candidate = self._build_candidate_text(item)
            if not candidate.strip():
                continue
            
            sim_score, sim_method = self.similarity.score(claim, candidate)
            desc_clean = self._clean_html(item.get('description', ''))

            stance = self._detect_stance(
                claim,
                item.get('title', ''),
                desc_clean
            )
            
            processed.append({
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'description': desc_clean,
                'published': item.get('published', ''),
                'source': source,
                'domain': item.get('domain', ''),
                'domain_tier': item.get('domain_tier', 5),
                'is_trusted': item.get('is_trusted', False),
                'credibility_score': item.get('credibility_score', 50),
                'claim_similarity': round(sim_score, 4),
                'similarity_method': sim_method,
                'stance': stance['stance'],
                'stance_confidence': stance['confidence'],
                'stance_detail': stance,
                'link_type': item.get('link_type', 'search'),
                'working_link': True
            })
        
        strict = [m for m in processed if m['claim_similarity'] >= SIMILARITY_THRESHOLD]
        if len(strict) >= 2:
            return strict
        
        return [m for m in processed if m['claim_similarity'] >= SIMILARITY_THRESHOLD_FALLBACK]
    
    def _calculate_match_score(self, matches: List[Dict], claim: str) -> int:
        """Calculate match score based on actual claim similarity + source credibility"""
        if not matches:
            return 0
        
        weighted_cred = sum(
            m.get('credibility_score', 50) * m.get('claim_similarity', 0)
            for m in matches
        )
        avg_sim = sum(m.get('claim_similarity', 0) for m in matches) / len(matches)
        avg_weighted = weighted_cred / len(matches)
        
        trusted_count = sum(
            1 for m in matches
            if m.get('is_trusted', False) and m.get('claim_similarity', 0) >= SIMILARITY_THRESHOLD
        )
        trusted_bonus = min(trusted_count * 10, 30)
        
        score = min(int(avg_weighted + trusted_bonus), 100)
        
        return score
    
    def _get_match_quality(self, score: int) -> str:
        """Get match quality description"""
        if score >= 85:
            return "HIGH CONFIDENCE - Verified by multiple trusted sources with strong claim alignment"
        elif score >= 70:
            return "MEDIUM CONFIDENCE - Verified by trusted sources with good claim alignment"
        elif score >= 50:
            return "MEDIUM CONFIDENCE - Multiple sources found but none are trusted or similarity is moderate"
        elif score >= 30:
            return "LOW CONFIDENCE - Some related articles with weak claim alignment"
        else:
            return "VERY LOW CONFIDENCE - No relevant matches"
    
    def _aggregate_source_stats(self, matches: List[Dict]) -> Dict:
        """Aggregate statistics about sources"""
        source_counts = {}
        trusted_sources = []
        unique_sources = []
        
        for match in matches:
            source = match.get('source', 'Unknown')
            domain = match.get('domain', '')
            is_trusted = match.get('is_trusted', False)
            
            if source not in source_counts:
                source_counts[source] = 0
                unique_sources.append(source)
            source_counts[source] += 1
            
            if is_trusted and source not in trusted_sources:
                trusted_sources.append(source)
        
        return {
            'total_sources': len(unique_sources),
            'trusted_count': len(trusted_sources),
            'unique_sources': unique_sources,
            'trusted_sources': trusted_sources,
            'source_counts': source_counts
        }
    
    def extract_source_from_url(self, url: str) -> str:
        """Extract source name from URL"""
        from urllib.parse import urlparse
        
        try:
            domain = urlparse(url).netloc
            domain = domain.replace('www.', '').lower()
            
            # Known source mappings
            source_map = {
                'bbc.com': 'BBC',
                'bbc.co.uk': 'BBC',
                'cnn.com': 'CNN',
                'reuters.com': 'Reuters',
            'apnews.com': 'AP News',
                'dawn.com': 'Dawn',
                'geo.tv': 'Geo News',
                'arynews.tv': 'ARY News',
                'samaa.tv': 'SAMAA',
                'thenews.com.pk': 'The News',
                'tribune.com.pk': 'Express Tribune',
                'theguardian.com': 'The Guardian',
                'nytimes.com': 'New York Times',
                'wsj.com': 'Wall Street Journal',
                'washingtonpost.com': 'Washington Post',
                'foxnews.com': 'Fox News',
                'dailymail.co.uk': 'Daily Mail',
                'usatoday.com': 'USA Today',
                'aljazeera.com': 'Al Jazeera',
                'dw.com': 'Deutsche Welle',
                'france24.com': 'France 24',
                'abcnews.go.com': 'ABC News',
                'nbcnews.com': 'NBC News',
                'cbsnews.com': 'CBS News',
                'skynews.com': 'Sky News',
                'independent.co.uk': 'The Independent',
                'telegraph.co.uk': 'The Telegraph',
                'timesofindia.indiatimes.com': 'Times of India',
                'hindustantimes.com': 'Hindustan Times',
            }
            
            for d, name in source_map.items():
                if d in domain:
                    return name
            
            # Return domain name if no match
            return domain.split('.')[0].title()
            
        except Exception:
            return 'Unknown'
    
    def get_credibility_score(self, domain: str) -> int:
        """Get credibility score for a domain via tier system"""
        info = resolve_domain(domain)
        return info.get('credibility_score', 50)


# For backward compatibility
TrustedSourcesDB = TrustedSourcesDB

