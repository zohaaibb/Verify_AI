 # app/services/orchestrator.py
import logging
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import re
import os
import time
from functools import wraps
import spacy
from collections import defaultdict
from threading import Lock
from urllib.parse import urlparse
import re
from app.services.text_processor import TextProcessor
from app.services.source_osint import SourceOSINT
from app.services.virustotal_client import VirusTotalClient
# from app.services.cuckoo_client import CuckooClient
# from app.services.reverse_engineering import ReverseEngineering
from app.services.image_forensics import ImageForensics
from app.services.trusted_sources import TrustedSourcesDB
from app.services.news_scraper import NewsScraper as NewsSiteScraper
from app.services.image_analysis_pipeline import ImageAnalysisPipeline
from app.services.url_scraper import URLScraper
# from app.services.sandbox_client import get_sandbox_orchestrator
from app.services.wikipedia_factcheck import WikipediaFactCheckModule
from app.services.url_ml_detector import URLMLDetector
from app.services.google_search_client import GoogleSearchClient

logger = logging.getLogger(__name__)

DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"


# ============================================
# Decorators
# ============================================

def input_type_guard(allowed_types):
    """Reject calls when the input_type is not applicable for this module."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, input_type, content):
            if input_type not in allowed_types:
                logger.warning(f"??  {func.__name__} not applicable for '{input_type}'")
                return {
                    'error':   f'{func.__name__} not applicable for {input_type}',
                    'success': False,
                    'module':  func.__name__.replace('run_', '').replace('_analysis', '')
                }
            return func(self, input_type, content)
        return wrapper
    return decorator


def timed_execution(func):
    """Log wall-clock time for every module execution."""
    @wraps(func)
    def wrapper(self, input_type, content):
        start = time.time()
        try:
            result  = func(self, input_type, content)
            elapsed = time.time() - start
            logger.info(f"??  {func.__name__} completed in {elapsed:.2f}s")
            if isinstance(result, dict):
                result['execution_time'] = round(elapsed, 2)
            return result
        except Exception:
            elapsed = time.time() - start
            logger.exception(f"?  {func.__name__} failed after {elapsed:.2f}s")
            raise
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry with exponential back-off; return error dict after all attempts."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, input_type, content):
            current_delay  = delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(self, input_type, content)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"??  {func.__name__} attempt {attempt + 1}/{max_retries} "
                                       f"failed, retrying in {current_delay:.1f}s")
                        time.sleep(current_delay)
                        current_delay *= backoff

            logger.error(f"?  {func.__name__} failed after {max_retries} attempts")
            return {
                'error':   str(last_exception),
                'success': False,
                'module':  func.__name__.replace('run_', '').replace('_analysis', '')
            }
        return wrapper
    return decorator


# ============================================
# Module Registry
# ============================================

class ModuleRegistry:
    """Thread-safe registry mapping module IDs ? functions / summarisers."""

    def __init__(self):
        self._modules       = {}
        self._summarizers   = {}
        self._applicability = defaultdict(list)
        self._lock          = Lock()

    def register(self, module_id: str, func,
                 summarizer=None, input_types: Optional[List[str]] = None):
        with self._lock:
            self._modules[module_id] = func
            if summarizer:
                self._summarizers[module_id] = summarizer
            for t in (input_types or []):
                if module_id not in self._applicability[t]:
                    self._applicability[t].append(module_id)

    def get_module(self, module_id: str):
        with self._lock:
            return self._modules.get(module_id)

    def get_summarizer(self, module_id: str):
        with self._lock:
            return self._summarizers.get(module_id)

    def get_applicable_modules(self, input_type: str, selected: List[str]) -> List[str]:
        with self._lock:
            valid = self._applicability.get(input_type, [])
        return [m for m in selected if m in valid]

    @property
    def all_modules(self) -> List[str]:
        with self._lock:
            return list(self._modules.keys())


# ============================================
# Standalone OSINT runner
# ============================================

def run_osint(osint_instance: SourceOSINT, mode: str, payload) -> dict:
    """
    Unified OSINT runner � called directly so mode-aware payloads can be
    passed without going through the (input_type, content: str) registry API.

    mode='text'  ? payload is a plain string
    mode='url'   ? payload is {url, domain, headline, body}
    mode='image' ? payload is a plain string (OCR text)
    """
    if mode == 'text':
        mentioned     = osint_instance.extract_sources_from_text(payload)
        verifications = [osint_instance.verify_source(s) for s in mentioned]
        overall = (
            sum(v['credibility_score'] for v in verifications) / len(verifications)
            if verifications else 0
        )
        return {
            'input_type':                'text',
            'mentioned_sources':         mentioned,
            'verifications':             verifications,
            'source_count':              len(mentioned),
            'overall_source_credibility': round(overall, 1),
        }

    elif mode == 'url':
        url      = payload.get('url') or payload.get('domain', '')
        headline = payload.get('headline', '')
        body     = payload.get('body', '')

        domain_result = osint_instance.get_credibility_for_url(url)

        article_text  = f"{headline} {body}".strip()
        mentioned     = osint_instance.extract_sources_from_text(article_text)
        verifications = [osint_instance.verify_source(s) for s in mentioned]
        overall = (
            sum(v['credibility_score'] for v in verifications) / len(verifications)
            if verifications else domain_result.get('credibility_score', 0)
        )
        return {
            'input_type':                'url',
            'domain_credibility':        domain_result,
            'mentioned_sources':         mentioned,
            'verifications':             verifications,
            'source_count':              len(mentioned),
            'overall_source_credibility': round(overall, 1),
        }

    elif mode == 'image':
        return osint_instance.analyse_ocr_text(payload)

    else:
        return {
            'input_type':        mode,
            'error':             f'Unknown OSINT mode: {mode}',
            'mentioned_sources': [],
            'verifications':     [],
            'source_count':      0,
        }


# ============================================
# Orchestrator
# ============================================

class AnalysisOrchestrator:
    """
    Central controller that routes analysis to the correct pipeline
    based on input type (text / url / image / file).
    """

    MAX_OSINT_VERIFICATIONS = 5
    SIMILARITY_THRESHOLD    = 0.45
    DEFAULT_TIMEOUT         = 30   # seconds, per non-scraper module
    SCRAPER_TIMEOUT         = 20   # scraper gets its own headroom
    URL_PIPELINE_TIMEOUT    = 120
    MAX_CONCURRENT_MODULES  = 4

    def __init__(self):
        self.text_processor  = TextProcessor()
        self.osint           = SourceOSINT()
        self.virustotal      = VirusTotalClient()
        # self.cuckoo = None
        # self.reverse = None
        self.forensics       = ImageForensics()
        self.trusted_sources = TrustedSourcesDB()
        self.news_scraper    = NewsSiteScraper(self.trusted_sources)
        self.image_pipeline  = ImageAnalysisPipeline()
        self.url_scraper     = URLScraper()
        self.wikipedia       = WikipediaFactCheckModule()
        self.url_ml_detector = URLMLDetector()
        self.google_search = GoogleSearchClient()

        # self.sandbox_orchestrator = None

        try:
            self.url_ml_detector.load()
        except Exception as e:
            logger.warning(f"?? URL ML detector not loaded: {e} � will use heuristic fallback")

        self.nlp = None   # spaCy lazy-loaded

        self.executor = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENT_MODULES)
        self._init_module_registry()
        logger.info(f"??  Orchestrator ready � modules: {self.registry.all_modules}")

    # ------------------------------------------------------------------
    # Registry
    # ------------------------------------------------------------------

    def _init_module_registry(self):
        self.registry = ModuleRegistry()

        self.registry.register('ai',         self.run_ai_analysis,
                               summarizer=self._summarize_ai,
                               input_types=['text', 'url', 'image'])
        self.registry.register('osint',      self.run_osint_analysis,
                               summarizer=self._summarize_osint,
                               input_types=['text', 'url', 'image'])
        self.registry.register('trusted',    self.run_trusted_sources_analysis,
                               summarizer=self._summarize_trusted,
                               input_types=['text', 'url', 'image'])
        self.registry.register('scraper',    self.run_direct_scraper_analysis,
                               summarizer=self._summarize_scraper,
                               input_types=['text', 'url'])
        self.registry.register('virustotal', self.run_virustotal_analysis,
                               summarizer=self._summarize_virustotal,
                               input_types=['url', 'file'])
        self.registry.register('vt',      self.run_virustotal_analysis,
                       summarizer=self._summarize_virustotal,
                       input_types=['url', 'file'])
        self.registry.register('urlml',  self.run_url_ml_analysis,
                       summarizer=self._summarize_url_ml,
                       input_types=['url'])
        # sandbox registration removed
                            #summarizer=self._summarize_sandbox,
                            ## summarizer=self._summarize_reverse,
                              # input_types=['file', 'url'])
        self.registry.register('wikipedia',  self.run_wikipedia_analysis,
                               summarizer=self._summarize_wikipedia,
                               input_types=['text'])
        self.registry.register('url_ml',    self.run_url_ml_analysis,
                               summarizer=self._summarize_url_ml,
                               input_types=['url'])
        self.registry.register('google_web_search', None,
                       summarizer=self._summarize_google_web,
                       input_types=['text', 'url', 'image'])
        self.registry.register('forensics',  self.run_forensics_analysis,
                       summarizer=self._summarize_forensics,
                       input_types=['image', 'file'])
        self.registry.register('deepfake',   self.run_deepfake_analysis,
                       summarizer=self._summarize_deepfake,
                       input_types=['image'])
    # ------------------------------------------------------------------
    # Top-level entry point (UPDATED for user_edited_text)
    # ------------------------------------------------------------------

    def analyze(self, input_type: str, content: str,
                selected_modules: List[str], **kwargs) -> Dict:
        logger.info(f"??  ANALYZE: input_type={input_type}, modules={selected_modules}")

        results = {
            'input_type':       input_type,
            'selected_modules': selected_modules,
            'module_results':   {},
            'summary':          {},
            'execution_stats':  {
                'total_time':        0,
                'modules_completed': 0,
                'modules_failed':    0,
            }
        }

        start_time    = time.time()
        valid_modules = []

        for module in selected_modules:
            if module not in self.registry.all_modules:
                logger.error(f"?  Unknown module: '{module}'")
                results['module_results'][module] = {
                    'error': f'Module {module} not available', 'success': False
                }
                results['execution_stats']['modules_failed'] += 1
            else:
                valid_modules.append(module)

        pipeline_map = {
            'url':   self.run_url_pipeline,
            'text':  self.run_text_pipeline,
            'image': self.run_image_pipeline,
            'file':  self.run_file_pipeline,
        }

        if input_type in pipeline_map:
            # -- Pass user_edited_text to image pipeline if provided --
            if input_type == 'image' and 'user_edited_text' in kwargs:
                pipeline_result = self.run_image_pipeline(
                    content, valid_modules,
                    user_edited_text=kwargs['user_edited_text']
                )
            else:
                pipeline_result = pipeline_map[input_type](content, valid_modules)
            results.update(pipeline_result)
        else:
            results['error'] = f"Unsupported input type: {input_type}"

        results['execution_stats']['total_time'] = round(time.time() - start_time, 2)
        return results

    # ------------------------------------------------------------------
    # Internal: run a dict of zero-arg callables in parallel
    # ------------------------------------------------------------------

    def _run_parallel(self, tasks: Dict[str, callable]) -> Dict[str, dict]:
        if not tasks:
            return {}

        results = {}
        with ThreadPoolExecutor(
            max_workers=min(self.MAX_CONCURRENT_MODULES, len(tasks))
        ) as ex:
            future_to_name = {ex.submit(fn): name for name, fn in tasks.items()}

            for future in as_completed(future_to_name):
                name    = future_to_name[future]
                timeout = self.SCRAPER_TIMEOUT if name == 'scraper' else self.DEFAULT_TIMEOUT
                try:
                    results[name] = future.result(timeout=timeout)
                    logger.info(f"?  {name} completed")
                except TimeoutError:
                    logger.warning(f"?  {name} timed out after {timeout}s � partial results kept")
                    results[name] = {
                        'error':   f'{name} timed out after {timeout}s',
                        'success': False,
                        'found':   False,
                    }
                except Exception as e:
                    logger.exception(f"?  {name} failed: {e}")
                    results[name] = {'error': str(e), 'success': False}

        return results

    # ------------------------------------------------------------------
    # TEXT PIPELINE
    # ------------------------------------------------------------------

    def run_text_pipeline(self, text: str, selected_modules: List[str]) -> Dict:
        results = {
            'input_type':       'text',
            'selected_modules': selected_modules,
            'module_results':   {},
            'summary':          {}
        }

        applicable = self.registry.get_applicable_modules('text', selected_modules)
        if not applicable:
            logger.warning(f"No applicable modules for text from {selected_modules}")
            return results

        clean_claim = self._clean_user_claim(text)

        logger.info(f"??  Text pipeline � modules: {applicable}")

        if 'ai' in applicable:
            try:
                logger.info(f"??  AI analysing claim: {clean_claim[:100]}�")
                ai_result = self.text_processor.analyze_text(clean_claim)
                results['module_results']['ai'] = ai_result
                logger.info(f"?  AI: is_fake={ai_result.get('is_fake')}, "
                             f"confidence={ai_result.get('confidence')}")
            except Exception as e:
                logger.exception(f"?  AI failed: {e}")
                results['module_results']['ai'] = {'error': str(e), 'success': False}

        parallel_tasks = {}

        if 'osint' in applicable:
            parallel_tasks['osint'] = lambda t=text: run_osint(self.osint, 'text', t)

        if 'trusted' in applicable:
            parallel_tasks['trusted'] = lambda t=clean_claim: \
                self.run_trusted_sources_analysis('text', t)

        if 'scraper' in applicable:
            parallel_tasks['scraper'] = lambda t=clean_claim: \
                self.run_direct_scraper_analysis('text', t)

        if 'wikipedia' in applicable:
            parallel_tasks['wikipedia'] = lambda t=clean_claim: \
                self.run_wikipedia_analysis('text', t)

                    
        if parallel_tasks:
            parallel_results = self._run_parallel(parallel_tasks)
            results['module_results'].update(parallel_results)

        # -- Google Web Search (fallback if Trusted Sources found nothing) --
        
        
        try:
            web_results = self.google_search.search_claim(text)
            results['module_results']['google_web_search'] = web_results
            logger.info(f"? Google Web Search: {web_results.get('total_results', 0)} results")
        except Exception as e:
            logger.error(f"? Google Web Search failed: {e}")


        results['summary'] = self._generate_combined_summary(results['module_results'], input_type='text')
        return results

    def _clean_user_claim(self, text: str) -> str:
        """Remove user-added source hints so AI/Trusted/Scraper get clean claim."""
        if not text:
            return text

        # Trailing source: " - BBC", " � dawn.com", " via Reuters"
        cleaned = re.sub(
            r'\s*(?:[-��]|\bvia\b)\s+\S+\.?\w*\s*$',
            '', text.strip(), flags=re.IGNORECASE
        ).strip()

        # Leading source: "BBC: ...", "bbc.com: ..."
        cleaned = re.sub(
            r'^[\w.-]+\s*:\s*',
            '', cleaned, flags=re.IGNORECASE
        ).strip()

        return cleaned if cleaned else text.strip()

    # ------------------------------------------------------------------
    # URL TYPE DETECTION
    # ------------------------------------------------------------------

    @staticmethod
    def detect_url_type(url: str) -> Dict[str, Any]:
        """
        Detect whether a URL points to a news ARTICLE or just a DOMAIN/BRAND homepage.

        Returns:
            {
                'type':       'article' | 'domain',
                'confidence': 0.0 - 1.0,
                'reason':     human-readable explanation,
                'matched_pattern': pattern that triggered the decision (or None),
            }
        """
        if not url:
            return {'type': 'domain', 'confidence': 1.0,
                    'reason': 'Empty URL', 'matched_pattern': None}

        # Article path indicators (strong signal)
        article_patterns = [
            (r'/news/',          'news section path'),
            (r'/story/',         'story path'),
            (r'/article/',       'article path'),
            (r'/articles/',      'articles path'),
            (r'/post/',          'post path'),
            (r'/posts/',         'posts path'),
            (r'/blog/',          'blog path'),
            (r'/politics/',      'politics section'),
            (r'/world/',         'world section'),
            (r'/business/',      'business section'),
            (r'/technology/',    'technology section'),
            (r'/tech/',          'tech section'),
            (r'/sports/',        'sports section'),
            (r'/entertainment/', 'entertainment section'),
            (r'/health/',        'health section'),
            (r'/science/',       'science section'),
            (r'/opinion/',       'opinion section'),
            (r'/culture/',       'culture section'),
            (r'/lifestyle/',     'lifestyle section'),
            (r'/[\d]{4}/[\d]{1,2}/[\d]{1,2}', 'date pattern YYYY/MM/DD'),
            (r'/[\d]{4}/[\d]{1,2}/',          'date pattern YYYY/MM'),
            (r'/[\d]{6,}\.html?$',            'numeric article ID .html'),
            (r'/[\d]{6,}\.php$',              'numeric article ID .php'),
            (r'/[\d]{6,}/?$',                 'numeric article ID'),
            (r'-\d{5,}/?$',                   'slug ending with numeric ID'),
        ]

        # Homepage / landing-page paths (strong domain signal)
        domain_paths = {
            '', '/', '/index.html', '/index.php', '/index.htm',
            '/home', '/home/', '/about', '/about/', '/about-us',
            '/about-us/', '/contact', '/contact/', '/contact-us',
            '/contact-us/', '/privacy', '/privacy/', '/terms',
            '/terms/', '/login', '/signin', '/signup', '/register'
        }

        try:
            parsed = urlparse(url if url.startswith(('http://', 'https://')) else f'http://{url}')
            path = (parsed.path or '').lower().rstrip('/')
            full_url_lower = url.lower()
        except Exception:
            return {'type': 'domain', 'confidence': 0.5,
                    'reason': 'URL parse failed', 'matched_pattern': None}

        # Rule 1: Empty or homepage path ? DOMAIN
        if path in domain_paths or path == '':
            return {
                'type': 'domain', 'confidence': 0.95,
                'reason': f"Path is homepage/landing ('{path or '/'}')",
                'matched_pattern': 'homepage_path'
            }

        # Rule 2: Match against article patterns ? ARTICLE
        for pattern, description in article_patterns:
            if re.search(pattern, full_url_lower):
                return {
                    'type': 'article', 'confidence': 0.90,
                    'reason': f'Matched article pattern: {description}',
                    'matched_pattern': pattern
                }

        # Rule 3: Heuristic � long descriptive slug suggests article
        path_parts = [p for p in path.split('/') if p]
        if path_parts:
            last_segment = path_parts[-1]
            if len(last_segment) > 20 and last_segment.count('-') >= 3:
                return {
                    'type': 'article', 'confidence': 0.80,
                    'reason': f"Long descriptive slug ({len(last_segment)} chars, "
                              f"{last_segment.count('-')} hyphens)",
                    'matched_pattern': 'long_slug_heuristic'
                }
            if len(path_parts) >= 3 and len(last_segment) > 10:
                return {
                    'type': 'article', 'confidence': 0.70,
                    'reason': f'Deep path with {len(path_parts)} segments',
                    'matched_pattern': 'deep_path_heuristic'
                }

        # Rule 4: Default � short/simple path ? DOMAIN
        return {
            'type': 'domain', 'confidence': 0.75,
            'reason': f"Path '{path}' is short/simple, no article indicators",
            'matched_pattern': None
        }

    # ------------------------------------------------------------------
    # URL PIPELINE (UPDATED: sends full article text to AI, trusted, scraper)
    # ------------------------------------------------------------------

    def run_url_pipeline(self, url: str, selected_modules: List[str]) -> Dict:
        """
        URL pipeline with automatic type detection.

        - DOMAIN/BRAND URL  ? lightweight pipeline (OSINT + VirusTotal only)
        - NEWS ARTICLE URL  ? full pipeline (now uses full article text for AI, trusted, scraper)
        """
        # -- Detect URL type FIRST -------------------------------------
        url_type_info = self.detect_url_type(url)
        url_type = url_type_info['type']

        logger.info(
            f"??  URL type detected: {url_type.upper()} "
            f"(confidence={url_type_info['confidence']:.2f}, "
            f"reason={url_type_info['reason']})"
        )

        # -- Branch: DOMAIN/BRAND URL ? lightweight pipeline -----------
        if url_type == 'domain':
            return self._run_domain_pipeline(url, selected_modules, url_type_info)

        # -- Branch: ARTICLE URL ? full pipeline with enhanced content --
        results = {
            'input_type':       'url',
            'url_type':         'article',
            'url_type_detection': url_type_info,
            'selected_modules': selected_modules,
            'module_results':   {},
            'summary':          {},
            'pipeline_stats':   {'phases': {}}
        }

        applicable = self.registry.get_applicable_modules('url', selected_modules)
        logger.info(f"??  URL pipeline (ARTICLE mode) � applicable modules: {applicable}")

        downloaded_files = []

        # -- Phase 1: VirusTotal + URL ML (parallel) --------------------
        phase_start      = time.time()
        is_malicious_url = False
        phase1_tasks     = {}

        if 'virustotal' in applicable:
            phase1_tasks['virustotal'] = lambda u=url: self.virustotal.check_url(u)

        if 'url_ml' in applicable:
            phase1_tasks['url_ml'] = lambda u=url: self.url_ml_detector.analyze(u)

        if phase1_tasks:
            phase1_results = self._run_parallel(phase1_tasks)
            results['module_results'].update(phase1_results)

            for key in ['virustotal', 'url_ml']:
                if key in phase1_results:
                    if phase1_results[key].get('verdict') == 'MALICIOUS':
                        is_malicious_url = True

        results['pipeline_stats']['phases']['virustotal_and_ml'] = round(time.time() - phase_start, 2)

        if is_malicious_url:
            results['early_exit'] = True
            results['summary'] = self._generate_combined_summary(results['module_results'], input_type='url')
            return results

        # -- Phase 2: Download check ------------------------------------
        phase_start = time.time()
        if False or 'sandbox' in applicable:
            try:
                download_check = self.url_scraper.check_for_downloads(url)
                if download_check.get('is_download') and download_check.get('success'):
                    file_path = download_check.get('file_path')
                    if file_path and os.path.exists(file_path):
                        downloaded_files.append(file_path)
                        if False:
                            results['module_results']['reverse'] = \
                                self.reverse.analyze_file(file_path)
                        if False:
                            results['module_results']['sandbox'] = \
                                self.sandbox_orchestrator.analyze_file(file_path)
            except Exception:
                logger.exception("?  Download check failed")

        results['pipeline_stats']['phases']['download_check'] = round(time.time() - phase_start, 2)

        # -- Phase 3: Scrape article (get headline and full text) -------
        phase_start    = time.time()
        scraped_result = {'success': False, 'headline': '', 'text': '', 'domain': ''}

        try:
            scraped_result = self.url_scraper.scrape_article(url)
            results['scraped_data'] = {
                'headline': scraped_result.get('headline', ''),
                'domain':   scraped_result.get('domain', ''),
                'url':      url,
            }
        except Exception as e:
            results['scraped_data'] = {'error': str(e)}

        results['pipeline_stats']['phases']['scrape'] = round(time.time() - phase_start, 2)

        headline = scraped_result.get('headline', '')
        body     = scraped_result.get('text', '')
        domain   = scraped_result.get('domain', '')

        # -- Use full article text for analysis (fallback to headline if body too short) --
        analysis_text = body if body and len(body.strip()) > 100 else headline
        logger.info(f"?? Using analysis text length: {len(analysis_text)} chars ({'body' if analysis_text == body else 'headline'})")

        # -- Phase 4: Parallel modules � now using full article text or headline --
        phase_start    = time.time()
        parallel_tasks = {}

        # AI � use full article text
        if 'ai' in applicable and analysis_text:
            parallel_tasks['ai'] = lambda t=analysis_text: self.run_ai_analysis('text', t)

        # OSINT � still uses URL mode with domain and article text
        if 'osint' in applicable:
            osint_payload = {
                'url':      url,
                'domain':   domain,
                'headline': headline,
                'body':     body,   # pass full body for better source extraction
            }
            parallel_tasks['osint'] = lambda p=osint_payload: run_osint(self.osint, 'url', p)

        # Trusted Sources � use full article text
        if 'trusted' in applicable and analysis_text:
            parallel_tasks['trusted'] = lambda t=analysis_text: self.run_trusted_sources_analysis('text', t)

        # Direct Scraper � use full article text (text mode) to search for similar news
        if 'scraper' in applicable and analysis_text:
            parallel_tasks['scraper'] = lambda t=analysis_text: self.run_direct_scraper_analysis('text', t)

        # Wikipedia � use full article text
        if 'wikipedia' in applicable and analysis_text:
            parallel_tasks['wikipedia'] = lambda t=analysis_text: self.run_wikipedia_analysis('text', t)

        if parallel_tasks:
            parallel_results = self._run_parallel(parallel_tasks)
            results['module_results'].update(parallel_results)

        results['pipeline_stats']['phases']['parallel'] = round(time.time() - phase_start, 2)

        # -- Cleanup ----------------------------------------------------
        for f in downloaded_files:
            try:
                if os.path.exists(f):
                    os.unlink(f)
            except Exception:
                pass

        results['summary'] = self._generate_combined_summary(results['module_results'], input_type='url')
        return results

    # ------------------------------------------------------------------
    # DOMAIN/BRAND URL PIPELINE (UPDATED: includes explicit URL ML call)
    # ------------------------------------------------------------------

    def _run_domain_pipeline(self, url: str, selected_modules: List[str],
                            url_type_info: Dict) -> Dict:
        """
        Lightweight pipeline for domain/brand URLs (e.g. google.com, bbc.com).

        Runs:
            - OSINT (domain credibility check)
            - VirusTotal (URL safety check)
            - URL ML (malicious/phishing detection) � explicitly called
        """
        logger.info(f"???   Domain/Brand pipeline for: {url}")

        results = {
            'input_type':         'url',
            'url_type':           'domain',
            'url_type_detection': url_type_info,
            'selected_modules':   selected_modules,
            'module_results':     {},
            'summary':            {},
            'pipeline_stats':     {'phases': {}},
            'notice': (
                "??  This is a domain/brand credibility check. "
                "For full news article analysis, please provide the complete "
                "article URL (e.g., https://example.com/news/article-title-12345)."
            ),
            'skipped_modules': [],
        }

        applicable = self.registry.get_applicable_modules('url', selected_modules)
        domain_applicable = [m for m in applicable if m in ('osint', 'virustotal', 'url_ml')]

        skipped = [m for m in applicable if m not in domain_applicable]
        if skipped:
            results['skipped_modules'] = skipped
            logger.info(f"??   Skipping (not applicable to domain URLs): {skipped}")

        if not domain_applicable:
            results['warnings'] = [
                'No applicable modules selected for domain analysis. '
                'Please select OSINT, VirusTotal, and/or URL ML.'
            ]
            results['summary'] = self._generate_combined_summary({})
            return results

        # Run OSINT and VirusTotal in parallel (if selected)
        phase_start = time.time()
        parallel_tasks = {}

        if 'osint' in domain_applicable:
            osint_payload = {
                'url':      url,
                'domain':   self.osint.extract_domain_from_url(url),
                'headline': '',
                'body':     '',
            }
            parallel_tasks['osint'] = lambda p=osint_payload: run_osint(self.osint, 'url', p)

        if 'virustotal' in domain_applicable:
            parallel_tasks['virustotal'] = lambda u=url: self.virustotal.check_url(u)

        if parallel_tasks:
            parallel_results = self._run_parallel(parallel_tasks)
            results['module_results'].update(parallel_results)

        # Explicitly run URL ML detector (bypass parallel issues)
        if 'url_ml' in domain_applicable:
            try:
                ml_result = self.url_ml_detector.analyze(url)
                results['module_results']['url_ml'] = ml_result
                logger.info(f"? URL ML: {ml_result.get('verdict')} (confidence {ml_result.get('confidence')})")
            except Exception as e:
                logger.error(f"? URL ML failed: {e}")
                results['module_results']['url_ml'] = {'error': str(e), 'success': False}

        results['pipeline_stats']['phases']['domain_checks'] = round(time.time() - phase_start, 2)

        # Build a domain-mode summary (includes URL ML if present)
        results['summary'] = self._generate_domain_summary(
            results['module_results'], url_type_info
        )

        logger.info(
            f"?  Domain pipeline complete in "
            f"{results['pipeline_stats']['phases']['domain_checks']}s"
        )
        return results

    def _generate_domain_summary(self, module_results: Dict, url_type_info: Dict) -> Dict:
        """
        Build a summary tailored for domain/brand checks.
        Different framing than the article summary.
        """
        summary = {
            'mode':                 'domain_check',
            'overall_threat_score': 0,
            'modules_completed':    len(module_results),
            'threats_detected':     [],
            'info':                 [],
            'recommendations':      [],
            'domain_assessment':    {
                'credibility_score': None,
                'credibility_level': 'UNKNOWN',
                'is_safe':           None,
            },
        }

        threat_score = 0.0

        # OSINT domain credibility
        osint = module_results.get('osint', {})
        if osint and not osint.get('error'):
            domain_cred = osint.get('domain_credibility', {})
            if domain_cred:
                cred_score = domain_cred.get('credibility_score', 0)
                cred_level = domain_cred.get('credibility_level', 'UNKNOWN')
                summary['domain_assessment']['credibility_score'] = cred_score
                summary['domain_assessment']['credibility_level'] = cred_level

                summary['info'].append({
                    'module':  'OSINT',
                    'message': f'Domain credibility: {cred_score}/100 ({cred_level})',
                    'details': {
                        'domain':           domain_cred.get('domain'),
                        'positive_signals': domain_cred.get('positive_signals', []),
                        'red_flags':        domain_cred.get('red_flags', []),
                        'ssl_valid':        domain_cred.get('ssl_valid', False),
                        'domain_age_years': domain_cred.get('domain_age_years', 0),
                    },
                })

                if cred_score < 30:
                    threat_score += 50
                    summary['threats_detected'].append({
                        'module':     'OSINT',
                        'threat':     f'Low domain credibility ({cred_score}/100)',
                        'confidence': round(100 - cred_score, 2),
                    })
                elif cred_score < 50:
                    threat_score += 25

        # VirusTotal safety
        vt = module_results.get('virustotal', {})
        if vt and not vt.get('error'):
            malicious = vt.get('malicious', 0)
            if malicious > 0:
                threat_score += vt.get('score', malicious * 10)
                summary['threats_detected'].append({
                    'module':     'VirusTotal',
                    'threat':     f'Flagged by {malicious} security vendors',
                    'confidence': round(vt.get('score', malicious * 10), 2),
                })
                summary['domain_assessment']['is_safe'] = False
            else:
                summary['domain_assessment']['is_safe'] = True
                summary['info'].append({
                    'module':  'VirusTotal',
                    'message': 'No security vendors flagged this domain',
                })

        # URL ML contribution (if present)
        url_ml = module_results.get('url_ml', {})
        if url_ml and not url_ml.get('error'):
            verdict = url_ml.get('verdict', 'UNKNOWN')
            if verdict == 'MALICIOUS':
                threat_score += url_ml.get('score', 80)
                summary['threats_detected'].append({
                    'module': 'URL Security',
                    'threat': f"Malicious URL detected ({url_ml.get('method', 'unknown')})",
                    'confidence': round(url_ml.get('confidence', 0) * 100, 2),
                    'message': url_ml.get('message', ''),
                })
            elif verdict == 'SUSPICIOUS':
                threat_score += url_ml.get('score', 45)
                summary['threats_detected'].append({
                    'module': 'URL Security',
                    'threat': f"Suspicious URL ({url_ml.get('method', 'unknown')})",
                    'confidence': round(url_ml.get('confidence', 0) * 100, 2),
                    'message': url_ml.get('message', ''),
                })
            else:
                summary['info'].append({
                    'module': 'URL Security',
                    'message': f"URL appears clean ({url_ml.get('method', 'unknown')})",
                })

        # Domain-specific recommendations
        summary['overall_threat_score'] = min(max(int(threat_score), 0), 100)
        score = summary['overall_threat_score']

        cred_score = summary['domain_assessment']['credibility_score']
        if score >= 70:
            summary['recommendations'].append(
                "?? HIGH RISK DOMAIN: Multiple red flags detected. "
                "Avoid interacting with this site."
            )
        elif score >= 40:
            summary['recommendations'].append(
                "??  CAUTION: This domain has some suspicious indicators. "
                "Verify legitimacy before sharing personal information."
            )
        elif cred_score is not None and cred_score >= 75:
            summary['recommendations'].append(
                f"? CREDIBLE DOMAIN: This domain shows strong legitimacy signals "
                f"({cred_score}/100)."
            )
        else:
            summary['recommendations'].append(
                "??  NEUTRAL: No major threats detected, but limited credibility data available."
            )

        summary['recommendations'].append(
            "?? To analyze a specific news article from this site, please submit "
            "the complete article URL."
        )

        summary['modules_succeeded'] = len([
            r for r in module_results.values()
            if not r.get('error') and r.get('success', True)
        ])
        summary['modules_failed'] = len([
            r for r in module_results.values()
            if r.get('error') or r.get('success') is False
        ])

        return summary

    # ------------------------------------------------------------------
    # IMAGE PIPELINE (UPDATED: user_edited_text support + URL extraction)
    # ------------------------------------------------------------------

    def run_image_pipeline(self, image_path: str, selected_modules: List[str],
                           user_edited_text: Optional[str] = None) -> Dict:
        results = {
            'input_type':       'image',
            'selected_modules': selected_modules,
            'module_results':   {},
            'summary':          {}
        }

        applicable = self.registry.get_applicable_modules('image', selected_modules)
        logger.info(f"???  Image pipeline � modules: {applicable}")

        extracted_text = ""

        # -- Phase 1: Forensics (includes OCR) ----------------------------
        if 'forensics' in applicable:
            try:
                forensics_result = self.image_pipeline.analyze_image(image_path, include_deepfake=False)
                results['module_results']['forensics'] = forensics_result
                extracted_text = forensics_result.get('ocr_results', {}).get('text', '').strip()
                word_count = len(extracted_text.split())
                logger.info(f"?? OCR extracted {word_count} words")
                if extracted_text:
                    logger.info(f"?? OCR Text preview: {extracted_text[:200]}")
            except Exception as e:
                logger.exception(f"? Forensics failed: {e}")
                results['module_results']['forensics'] = {'error': str(e), 'success': False}

        # Fallback OCR
        if not extracted_text:
            try:
                extracted_text = self.image_pipeline.get_text_for_analysis(image_path)
                extracted_text = (extracted_text or "").strip()
                if extracted_text:
                    logger.info(f"?? Fallback OCR: {len(extracted_text.split())} words")
            except Exception as e:
                logger.warning(f"?? Fallback OCR failed: {e}")

        # OCR Garbage cleaner (strict filter)
        if extracted_text:
            lines = extracted_text.split('\n')
            clean_lines = []
            for line in lines:
                words = line.split()
                if not words:
                    continue
                alpha_words = [w for w in words if w.isalpha()]
                if len(alpha_words) < 3:                     # too few actual words
                    continue
                avg_len = sum(len(w) for w in alpha_words) / len(alpha_words)
                if avg_len < 3.5:                            # too short on average
                    continue
                clean_lines.append(line)
            if clean_lines:
                extracted_text = '\n'.join(clean_lines).strip()
                logger.info(f"?? Cleaned OCR: {len(extracted_text.split())} words (strict filter)")
            else:
                extracted_text = ""

        # ?? USER-EDITED TEXT OVERRIDE (agar frontend ne bheja ho)
        if user_edited_text:
            user_edited_text = user_edited_text.strip()
            if user_edited_text:
                extracted_text = user_edited_text
                logger.info(f"?? Using user-edited text ({len(extracted_text.split())} words): {extracted_text[:200]}")

        # -- Phase 2: Run AI, OSINT, Trusted Sources, Scraper (uses extracted_text) --
        if extracted_text and len(extracted_text.split()) >= 3:
            logger.info(f"?? Running text analysis on {len(extracted_text.split())} words")
            
            if 'ai' in applicable:
                try:
                    ai_result = self.text_processor.analyze_text(extracted_text)
                    ai_result['verdict'] = 'REAL' if not ai_result.get('is_fake') else 'FAKE'
                    results['module_results']['ai'] = ai_result
                    logger.info(f"? AI: {ai_result['verdict']} ({ai_result.get('confidence')}%)")
                except Exception as e:
                    logger.error(f"? AI failed: {e}")
                    results['module_results']['ai'] = {'error': str(e)}
            
            if 'osint' in applicable:
                try:
                    osint_result = self.osint.analyse_ocr_text(extracted_text)
                    results['module_results']['osint'] = osint_result
                    logger.info(f"? OSINT: {len(osint_result.get('mentioned_sources', []))} sources")
                except Exception as e:
                    logger.error(f"? OSINT failed: {e}")
                    results['module_results']['osint'] = {'error': str(e)}
            
            if 'trusted' in applicable:
                try:
                    trusted_result = self.trusted_sources.check_claim_against_trusted(extracted_text)
                    results['module_results']['trusted'] = trusted_result
                    logger.info(f"? Trusted Sources: verified={trusted_result.get('verified', False)}")
                except Exception as e:
                    logger.error(f"? Trusted Sources failed: {e}")
                    results['module_results']['trusted'] = {'error': str(e)}
            
            # -- Direct Scraper for images (bypass registry) ---------------
            if 'scraper' in selected_modules:
                try:
                    scraper_result = self.news_scraper.verify_news_exists(extracted_text)
                    results['module_results']['scraper'] = scraper_result
                    logger.info(f"? Scraper: found={scraper_result.get('found', False)}")
                except Exception as e:
                    logger.error(f"? Scraper failed: {e}")
                    results['module_results']['scraper'] = {'error': str(e)}

            # -- Phase 2.5: Analyze any URLs found in OCR text -------------
            urls = self._extract_urls(extracted_text)
            if urls:
                target_url = urls[0]  # use the first URL found
                logger.info(f"?? Found URL in image text: {target_url}")

                # URL ML Detection (if selected)
                if 'url_ml' in selected_modules:
                    try:
                        url_ml_result = self.url_ml_detector.analyze(target_url)
                        results['module_results']['url_ml'] = url_ml_result
                        logger.info(f"? URL ML: {url_ml_result.get('verdict')} (confidence {url_ml_result.get('confidence')})")
                    except Exception as e:
                        logger.error(f"? URL ML failed on image URL: {e}")
                        results['module_results']['url_ml'] = {'error': str(e)}

                # VirusTotal URL check (if selected)
                if 'virustotal' in selected_modules:
                    try:
                        vt_result = self.virustotal.check_url(target_url)
                        results['module_results']['virustotal'] = vt_result
                        logger.info(f"? VirusTotal URL: malicious={vt_result.get('malicious', 0)}")
                    except Exception as e:
                        logger.error(f"? VirusTotal URL check failed: {e}")
                        results['module_results']['virustotal'] = {'error': str(e)}

        else:
            logger.warning(f"?? OCR text too short, skipping AI/OSINT")

        # -- Phase 3: Deepfake ---------------------------------------------
        if 'deepfake' in applicable:
            try:
                results['module_results']['deepfake'] = self.run_deepfake_analysis('image', image_path)
            except Exception as e:
                results['module_results']['deepfake'] = {'error': str(e)}

        # -- Phase 4: VirusTotal (image file scan) -------------------------
        # Already handled above; if virustotal was also selected, the URL check
        # will be present. The image file scan can run as well if needed.
        # (existing logic already calls check_file, we keep both)
        if 'virustotal' in applicable and 'virustotal' not in [m for m in selected_modules if m == 'virustotal']:
            try:
                results['module_results']['virustotal'] = self.virustotal.check_file(image_path)
            except Exception as e:
                results['module_results']['virustotal'] = {'error': str(e)}

        results['summary'] = self._generate_combined_summary(results['module_results'], input_type='image')
        return results

    def _extract_urls(self, text: str) -> List[str]:
        """Return a list of valid HTTP(S) URLs found in text."""
        if not text:
            return []
        url_pattern = r'https?://[^\s()<>"\']+'
        urls = re.findall(url_pattern, text, re.IGNORECASE)
        # remove duplicates while preserving order
        seen = set()
        unique = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                unique.append(u)
        return unique

    # ------------------------------------------------------------------
    # FILE PIPELINE (unchanged)
    # ------------------------------------------------------------------

    def run_file_pipeline(self, file_path: str, selected_modules: List[str]) -> Dict:
        results = {
            'input_type':       'file',
            'selected_modules': selected_modules,
            'module_results':   {},
            'summary':          {}
        }

        applicable = self.registry.get_applicable_modules('file', selected_modules)
        logger.info(f"??  File pipeline � modules: {applicable}")

        if 'forensics' in applicable:
            try:
                results['module_results']['forensics'] = self.forensics.analyze_file(file_path)
            except Exception as e:
                results['module_results']['forensics'] = {'error': str(e), 'success': False}

        if False:
            try:
                results['module_results']['reverse'] = self.reverse.analyze_file(file_path)
            except Exception as e:
                results['module_results']['reverse'] = {'error': str(e), 'success': False}

        if False:
            try:
                results['module_results']['sandbox'] = \
                    self.sandbox_orchestrator.analyze_file(file_path)
            except Exception as e:
                results['module_results']['sandbox'] = {'error': str(e), 'success': False}

        if 'virustotal' in applicable:
            try:
                results['module_results']['virustotal'] = self.virustotal.check_file(file_path)
            except Exception as e:
                results['module_results']['virustotal'] = {'error': str(e), 'success': False}

        results['summary'] = self._generate_combined_summary(results['module_results'], input_type='file')
        return results

    # ------------------------------------------------------------------
    # Module wrappers (unchanged)
    # ------------------------------------------------------------------

    @input_type_guard(['text'])
    @timed_execution
    @retry_on_failure(max_retries=2)
    def run_ai_analysis(self, input_type: str, content: str) -> Dict:
        if not content or len(content.strip()) < 10:
            return {'success': False, 'error': 'Text too short', 'is_fake': False, 'confidence': 0}
        return self.text_processor.analyze_text(content)

    @input_type_guard(['text', 'url', 'image'])
    @timed_execution
    def run_osint_analysis(self, input_type: str, content: str) -> Dict:
        return run_osint(self.osint, input_type, content)

    @input_type_guard(['text', 'url', 'image'])
    @timed_execution
    @retry_on_failure(max_retries=2)
    def run_trusted_sources_analysis(self, input_type: str, content: str) -> Dict:
        return self.trusted_sources.check_claim_against_trusted(content)

    @input_type_guard(['text', 'url'])
    @timed_execution
    @retry_on_failure(max_retries=2)
    def run_direct_scraper_analysis(self, input_type: str, content: str) -> Dict:
        if input_type == 'url':
            return self.url_scraper.scrape_article(content)
        return self.news_scraper.verify_news_exists(content)

    @input_type_guard(['url', 'file'])
    @timed_execution
    @retry_on_failure(max_retries=2)
    def run_virustotal_analysis(self, input_type: str, content: str) -> Dict:
        if input_type == 'url':
            return self.virustotal.check_url(content)
        return self.virustotal.check_file(content)

    @input_type_guard(['url', 'file', 'image'])
    @timed_execution
    def _disabled_sandbox(self, input_type: str, content: str) -> Dict:
        if input_type == 'file':
            if not os.path.exists(content):
                return {'error': f'File not found: {content}', 'success': False}
            result = self.sandbox_orchestrator.analyze_file(content)
            if result.get('success'):
                summary = result.get('summary', {})
                if summary.get('malicious'):
                    if hasattr(self, 'summary'):
                        self.summary['threats_detected'].append(
                            f"Sandbox detection: {summary.get('threat_level')} threat detected"
                        )
            return result
        elif input_type == 'url':
            return self.virustotal.check_url(content)
        return {'error': 'Sandbox not applicable', 'success': False}

    @input_type_guard(['image'])
    @timed_execution
    def run_deepfake_analysis(self, input_type: str, content: str) -> Dict:
        try:
            result = self.image_pipeline.analyze_image(content, include_deepfake=True)
            if result.get('success'):
                return {
                    'success':         True,
                    'deepfake_result': result.get('deepfake_results', {}).get('deepfake_result', {}),
                    'image_info':      result.get('image_info', {}),
                    'ocr_text':        result.get('ocr_results', {}).get('text', ''),
                }
            return {'success': False, 'error': result.get('error', 'Deepfake analysis failed')}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @input_type_guard(['image', 'file'])
    @timed_execution
    def run_forensics_analysis(self, input_type: str, content: str) -> Dict:
        if input_type == 'image':
            return self.image_pipeline.analyze_image(content)
        return self.forensics.analyze_file(content)

    @input_type_guard(['file', 'url'])
    @timed_execution
    @retry_on_failure(max_retries=1)
    def _disabled_reverse(self, input_type: str, content: str) -> Dict:
        if input_type == 'url':
            return {'success': False, 'error': 'Reverse analysis requires file path, not URL'}
        if not os.path.exists(content):
            return {'success': False, 'error': f'File not found: {content}'}
        return self.reverse.analyze_file(content)

    @input_type_guard(['text'])
    @timed_execution
    def run_wikipedia_analysis(self, input_type: str, content: str) -> Dict:
        return self.wikipedia.analyze(content)

    @input_type_guard(['url'])
    @timed_execution
    def run_url_ml_analysis(self, input_type: str, content: str) -> Dict:
        if not content:
            return {'success': False, 'error': 'No URL provided'}
        return self.url_ml_detector.analyze(content)

    # ================================================================== #
    # SCORING SYSTEM (bi-directional, starts at 50)
    # ================================================================== #

    def _confidence_scale(self, points: float, confidence: float) -> float:
        if confidence >= 0.80:
            return points
        elif confidence >= 0.60:
            return points * 0.5
        else:
            return points * 0.25

    def _summarize_ai(self, result: Dict, summary: Dict, score: float) -> float:
        is_fake = result.get('is_fake', False)
        confidence = result.get('confidence', 0)
        input_type = summary.get('input_type', 'text')

        if is_fake:
            base = 35 if confidence >= 0.80 else 20
            delta = +self._confidence_scale(base, confidence)
            label = f"AI FAKE (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'AI Analysis',
                'threat': 'Fake News Detected',
                'confidence': round(confidence * 100, 2)
            })
        else:
            # REAL � stronger penalty for images per user request
            if input_type == 'image':
                base = 30 if confidence >= 0.80 else 15
            else:
                base = 20 if confidence >= 0.80 else 10
            delta = -self._confidence_scale(base, confidence)
            label = f"AI REAL ({delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'AI Analysis',
                'message': f'AI identifies this as REAL news ({confidence*100:.1f}% confidence)'
            })
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_osint(self, result: Dict, summary: Dict, score: float) -> float:
        delta = 0.0
        domain_cred = result.get('domain_credibility')
        if domain_cred:
            summary.setdefault('info', []).append({
                'module': 'OSINT',
                'message': (f"Domain credibility: {domain_cred.get('credibility_score', 0)}/100 "
                           f"({domain_cred.get('credibility_level', 'UNKNOWN')})"),
                'domain': domain_cred.get('domain', ''),
                'ssl_valid': domain_cred.get('ssl_valid', False),
                'domain_age': domain_cred.get('domain_age_years', 0),
            })
        for src in result.get('mentioned_sources', []):
            summary.setdefault('info', []).append({
                'module': 'OSINT', 'message': f"Mentioned source: {src}"
            })
        overall = result.get('overall_source_credibility', 50)
        if overall >= 80:
            delta = -15
            label = f"OSINT HIGH credibility ({delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'OSINT',
                'message': f"Sources have high credibility ({overall:.0f}/100)",
                'confidence': overall,
            })
        elif overall >= 60:
            delta = -5
            label = f"OSINT MEDIUM credibility ({delta:.0f})"
        elif overall >= 50:
            delta = 0
            label = f"OSINT NEUTRAL credibility ({delta:.0f})"
        elif overall >= 30:
            delta = +15
            label = f"OSINT LOW credibility (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'OSINT',
                'threat': 'Low Source Credibility',
                'confidence': round(100 - overall, 2)
            })
        else:
            delta = +25
            label = f"OSINT VERY LOW credibility (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'OSINT',
                'threat': 'Unreliable Sources Detected',
                'confidence': round(100 - overall, 2)
            })
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_trusted(self, result: Dict, summary: Dict, score: float) -> float:
        matches = result.get('matches', [])
        n = len(matches)
        match_score = result.get('match_score', 0)
        verified = result.get('verified', False)

        if verified and match_score >= 80:
            delta = -25
            label = f"Trusted Sources VERIFIED ({delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'Trusted Sources',
                'message': f'Verified by {n} trusted sources (match score {match_score}/100)'
            })
        elif n >= 3 and match_score >= 60:
            delta = -10
            label = f"Trusted Sources RELATED COVERAGE ({delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'Trusted Sources',
                'message': f'Found {n} related articles (match score {match_score}/100) � not a direct match'
            })
        elif n > 0:
            delta = 0
            label = f"Trusted Sources {n} LOW-RELEVANCE matches (0)"
            summary.setdefault('info', []).append({
                'module': 'Trusted Sources',
                'message': f'Found {n} loosely related articles � insufficient to verify'
            })
        else:
            delta = +15
            label = f"Trusted Sources NO MATCHES (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Trusted Sources',
                'threat': 'No matching trusted sources found',
                'confidence': 70
            })

        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_scraper(self, result: Dict, summary: Dict, score: float) -> float:
        matches_found = result.get('matches_found', 0)
        found = result.get('found', False)

        if found and matches_found >= 3:
            delta = -15
            label = f"Scraper FOUND {matches_found} matches ({delta:.0f})"
        elif found and matches_found >= 1:
            delta = -5
            label = f"Scraper FOUND {matches_found} match(es) ({delta:.0f})"
        else:
            delta = +10
            label = f"Scraper NOT FOUND (+{delta:.0f})"

        if matches_found > 0:
            summary.setdefault('info', []).append({
                'module': 'Direct Scraper',
                'message': f'Found {matches_found} articles. They may contain related but not identical claims.'
            })

        summary.setdefault('score_influences', []).append(label)
        return score + delta
    
    def _summarize_virustotal(self, result: Dict, summary: Dict, score: float) -> float:
        malicious = result.get('malicious', 0)
        if malicious > 0:
            delta = +30
            label = f"VirusTotal MALICIOUS {malicious} vendors (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'VirusTotal',
                'threat': f"Flagged by {malicious} security vendors",
                'confidence': round(result.get('score', malicious * 10), 2)
            })
        else:
            delta = 0
            label = "VirusTotal CLEAN (0)"
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_url_ml(self, result: Dict, summary: Dict, score: float) -> float:
        verdict = result.get('verdict', 'CLEAN')
        confidence = result.get('confidence', 0.5)
        method = result.get('method', 'unknown')
        if verdict in ('MALICIOUS', 'SUSPICIOUS'):
            base = 30 if verdict == 'MALICIOUS' else 15
            delta = +self._confidence_scale(base, confidence)
            label = f"URL ML {verdict} (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'URL ML Detection',
                'threat': f"ML detected: {result.get('classification', verdict)} ({method})",
                'confidence': round(confidence * 100, 2),
                'score': result.get('score', 0)
            })
        else:
            delta = 0
            label = f"URL ML CLEAN ({method}) (0)"
            summary.setdefault('info', []).append({
                'module': 'URL ML Detection',
                'message': f"URL classified as {result.get('classification', 'CLEAN')}",
                'method': method
            })
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_sandbox(self, result: Dict, summary: Dict, score: float) -> float:
        summary_data = result.get('summary', {})
        threat_level = summary_data.get('threat_level', 'LOW')
        if summary_data.get('malicious') or threat_level in ('HIGH', 'MEDIUM'):
            threat_score = summary_data.get('threat_score', 70)
            delta = +min(threat_score, 40)
            label = f"Sandbox {threat_level} (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Sandbox',
                'threat': f"Malicious Behavior Detected ({threat_level})",
                'confidence': round(threat_score, 2)
            })
        else:
            delta = 0
            label = "Sandbox CLEAN (0)"
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_reverse(self, result: Dict, summary: Dict, score: float) -> float:
        if not result.get('success', False):
            return score
        verdict = result.get('verdict', {})
        if verdict.get('is_malicious'):
            threat_score = verdict.get('threat_score', 70)
            delta = +min(threat_score, 35)
            label = f"Reverse Engineering MALICIOUS (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Reverse Engineering',
                'threat': f"Malicious Code Patterns ({verdict.get('threat_level', 'HIGH')})",
                'confidence': round(threat_score, 2)
            })
            summary.setdefault('score_influences', []).append(label)
            return score + delta
        return score

    def _summarize_forensics(self, result: Dict, summary: Dict, score: float) -> float:
        hidden = result.get('hidden_payload', {})
        if hidden.get('has_payload'):
            delta = +25
            label = f"Forensics HIDDEN PAYLOAD (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Forensics',
                'threat': 'Hidden Payload Detected',
                'confidence': 90.0
            })
        else:
            delta = 0
            label = "Forensics CLEAN (0)"
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_deepfake(self, result: Dict, summary: Dict, score: float) -> float:
        deepfake = result.get('deepfake_result', {})
        confidence = deepfake.get('confidence', 0.7)
        if deepfake.get('is_deepfake'):
            delta = +self._confidence_scale(30, confidence)
            label = f"Deepfake DETECTED (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Deepfake',
                'threat': 'AI-manipulated Image Detected',
                'confidence': round(confidence * 100, 2)
            })
        else:
            delta = 0
            label = "Deepfake CLEAN (0)"
        summary.setdefault('score_influences', []).append(label)
        return score + delta

    def _summarize_wikipedia(self, result: Dict, summary: Dict, score: float) -> float:
        if result.get('verified'):
            delta = -10
            label = f"Wikipedia VERIFIED ({delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'Wikipedia Fact-Check',
                'message': f"Verified by Wikipedia: {result.get('verdict')}",
                'confidence': round(result.get('confidence', 0) * 100, 2),
                'url': result.get('wikipedia_url', '')
            })
        else:
            delta = +5
            label = f"Wikipedia NOT VERIFIED (+{delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'Wikipedia Fact-Check',
                'message': f"Not found in Wikipedia � {result.get('verdict')}",
                'note': 'Claim could not be verified against Wikipedia'
            })
        summary.setdefault('score_influences', []).append(label)
        return score + delta
    def _summarize_google_web(self, result: Dict, summary: Dict, score: float) -> float:
        social = result.get('social_media_sources', [])
        credible = result.get('credible_sources', [])
        total = result.get('total_results', 0)

        if total == 0:
            delta = +20
            label = f"Google Web Search NO RESULTS (+{delta:.0f})"
            summary['threats_detected'].append({
                'module': 'Google Web Search',
                'threat': 'Claim not found anywhere on the web',
                'confidence': 85
            })
        elif social and not credible:
            delta = +10
            label = f"Google Web Search SOCIAL MEDIA ONLY (+{delta:.0f})"
            summary.setdefault('info', []).append({
                'module': 'Google Web Search',
                'message': f'Found only on social media ({len(social)} results) � no trusted news coverage'
            })
        elif credible:
            delta = -10
            label = f"Google Web Search CREDIBLE SOURCES ({delta:.0f})"
        else:
            delta = 0
            label = "Google Web Search NEUTRAL (0)"

        summary.setdefault('score_influences', []).append(label)
        return score + delta
    # ------------------------------------------------------------------
    # COMBINED SUMMARY � starts at 50, now accepts input_type
    # ------------------------------------------------------------------

    def _generate_combined_summary(self, module_results: Dict, input_type: Optional[str] = None) -> Dict:
        summary = {
            'overall_threat_score': 0,
            'modules_completed':    len(module_results),
            'threats_detected':     [],
            'info':                 [],
            'recommendations':      [],
            'score_influences':     [],
            'input_type':           input_type,
        }
        score = 50.0
        for module, result in module_results.items():
            summarizer = self.registry.get_summarizer(module)
            if summarizer and result and not result.get('error'):
                try:
                    score = summarizer(result, summary, score)
                except Exception as e:
                    logger.exception(f"Error in summarizer for {module}: {e}")
        final = int(max(0, min(100, score)))
        summary['overall_threat_score'] = final
        summary['score_influences'].append(f"Started: 50 ? Final: {final}")
        if final <= 15:
            summary['verdict'] = 'SAFE'
            summary['recommendations'].append("? SAFE: Content appears legitimate and credible.")
        elif final <= 40:
            summary['verdict'] = 'LOW THREAT'
            summary['recommendations'].append("?? LOW THREAT: Some indicators present. Exercise normal caution.")
        elif final <= 70:
            summary['verdict'] = 'MEDIUM THREAT'
            summary['recommendations'].append("?? MEDIUM THREAT: Multiple suspicious indicators detected. Verify through trusted sources before sharing.")
        else:
            summary['verdict'] = 'HIGH THREAT'
            summary['recommendations'].append("?? HIGH THREAT: Content appears malicious or fake. Do not trust or share.")
        summary['modules_succeeded'] = len([r for r in module_results.values() if not r.get('error') and r.get('success', True)])
        summary['modules_failed'] = len([r for r in module_results.values() if r.get('error') or r.get('success') is False])
        return summary

    def _extract_source_names(self, text: str) -> List[str]:
        """Extract source names, using spaCy NER if available, else simple fallback."""

        # Lazy-load spaCy if not already attempted
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("? spaCy model lazy-loaded")
            except Exception as e:
                logger.warning(f"spaCy not available ({e}), using simple extraction")
                self.nlp = False   # mark as failed � never try again

        # If spaCy loaded successfully, use it
        if self.nlp and self.nlp is not False:
            doc = self.nlp(text)
            sources = []
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PERSON", "GPE"]:
                    if any(w in ent.text.lower() for w in ('news', 'times', 'post', 'tribune', 'tv', 'channel')):
                        sources.append(ent.text)
            # Also use built-in list + patterns for extra coverage
            news_sources = [
                'BBC', 'CNN', 'Reuters', 'AP', 'AFP', 'Al Jazeera',
                'Dawn', 'Geo', 'ARY', 'SAMAA', 'Express Tribune', 'The News',
                'NDTV', 'Times of India', 'The Hindu', 'Hindustan Times',
                'Guardian', 'The Guardian', 'NYT', 'New York Times',
                'Washington Post', 'Wall Street Journal', 'WSJ',
                'Fox News', 'NBC', 'ABC', 'CBS', 'MSNBC', 'Sky News',
                'ITV', 'Channel 4', 'Bloomberg', 'Forbes', 'Business Insider'
            ]
            text_lower = text.lower()
            for s in news_sources:
                if s.lower() in text_lower and s not in sources:
                    sources.append(s)
            patterns = [
                r'according to ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'reported by ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
                r'via ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
            ]
            for pattern in patterns:
                for match in re.findall(pattern, text):
                    if (len(match) > 2 and match not in sources and
                            any(w in match.lower() for w in ('news', 'times', 'post', 'tribune', 'tv', 'channel'))):
                        sources.append(match)
            return list(dict.fromkeys(sources))

        # Fallback: simple extraction without spaCy
        return self._simple_source_names(text)

    def _simple_source_names(self, text: str) -> List[str]:
        """Simple source extraction without spaCy."""
        news_sources = [
            'BBC', 'CNN', 'Reuters', 'AP', 'AFP', 'Al Jazeera',
            'Dawn', 'Geo', 'ARY', 'SAMAA', 'Express Tribune', 'The News',
            'NDTV', 'Times of India', 'The Hindu', 'Hindustan Times',
            'Guardian', 'The Guardian', 'NYT', 'New York Times',
            'Washington Post', 'Wall Street Journal', 'WSJ',
            'Fox News', 'NBC', 'ABC', 'CBS', 'MSNBC', 'Sky News',
            'ITV', 'Channel 4', 'Bloomberg', 'Forbes', 'Business Insider'
        ]
        text_lower = text.lower()
        sources = [s for s in news_sources if s.lower() in text_lower]

        # Also catch domain-like patterns
        domain_pattern = r'\b[\w.-]+\.[a-z]{2,}\b'
        for domain in re.findall(domain_pattern, text, re.IGNORECASE):
            if domain.lower() not in [s.lower() for s in sources]:
                sources.append(domain)

        return list(dict.fromkeys(sources))

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
