# app/services/url_scraper.py

import logging
import requests
import time
import random
import tempfile
import os
import re
from urllib.parse import urlparse, quote
from bs4 import BeautifulSoup

# Try to import cloudscraper for bypassing Cloudflare
try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# Try to import playwright for headless browser
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Try to import newspaper for article extraction
try:
    from newspaper import Article, Config as NewspaperConfig
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================
# USER AGENTS POOL (Rotating)
# ============================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0",
]


def _build_headers(url: str) -> dict:
    """Build realistic headers to avoid blocking"""
    domain = urlparse(url).netloc
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": f"https://www.google.com/search?q={quote(domain)}",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "cross-site",
        "Cache-Control": "max-age=0",
    }


def _fetch_with_playwright(url: str) -> str | None:
    """Fetch using Playwright headless browser (bypasses Cloudflare)"""
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-US",
                viewport={"width": 1280, "height": 800},
            )
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20000)
            time.sleep(2)  # Let JavaScript render
            html = page.content()
            browser.close()
            return html
    except Exception as e:
        logger.warning(f"[Playwright] fetch failed: {e}")
        return None


def _fetch_from_archive(url: str) -> str | None:
    """Fetch cached copy from Wayback Machine"""
    try:
        api = f"https://archive.org/wayback/available?url={quote(url)}"
        response = requests.get(api, timeout=8)
        data = response.json()
        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available"):
            cached_url = snapshot["url"]
            headers = _build_headers(cached_url)
            r2 = requests.get(cached_url, headers=headers, timeout=15)
            if r2.status_code == 200:
                logger.info(f"[Archive] Retrieved cached version from {cached_url[:80]}")
                return r2.text
    except Exception as e:
        logger.warning(f"[Archive] fetch failed: {e}")
    return None


def _extract_article(html: str, url: str) -> dict:
    """Extract title, text, authors from raw HTML using multiple methods"""
    
    # ============================================
    # METHOD 1: Try newspaper3k with HTML input
    # ============================================
    if NEWSPAPER_AVAILABLE:
        try:
            config = NewspaperConfig()
            config.browser_user_agent = random.choice(USER_AGENTS)
            config.request_timeout = 15
            
            article = Article(url, config=config)
            article.html = html          # ✅ FIX: provide raw HTML directly
            article.parse()
            
            if article.title and len(article.title) > 5:
                logger.debug(f"Newspaper extraction successful: {article.title[:50]}...")
                return {
                    "title": article.title or "",
                    "text": article.text or "",
                    "authors": article.authors or [],
                    "top_image": article.top_image or "",
                    "publish_date": str(article.publish_date) if article.publish_date else "",
                }
        except Exception as e:
            logger.debug(f"Newspaper extraction failed: {e}")
    
    # ============================================
    # METHOD 2: BeautifulSoup fallback
    # ============================================
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "form"]):
            tag.decompose()
        
        # ============================================
        # Extract Title
        # ============================================
        title = ""
        
        # Try OpenGraph title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        
        # Try Twitter title
        if not title:
            twitter_title = soup.find("meta", property="twitter:title")
            if twitter_title and twitter_title.get("content"):
                title = twitter_title["content"]
        
        # Try H1
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)
        
        # Try page title
        if not title and soup.title:
            title = soup.title.string
        
        # Clean title
        if title:
            title = re.sub(r'\s+', ' ', title).strip()
        
        # ============================================
        # Extract Text Content
        # ============================================
        text = ""
        
        # Try article tag
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
            text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
        
        # Try main content area
        if not text:
            main_selectors = ["main", "#main", ".main-content", ".article-content", ".story-content", ".entry-content"]
            for selector in main_selectors:
                main_tag = soup.select_one(selector)
                if main_tag:
                    paragraphs = main_tag.find_all("p")
                    text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 30])
                    if text:
                        break
        
        # Fallback: all paragraphs
        if not text:
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50])
        
        # Clean text
        if text:
            text = re.sub(r'\s+', ' ', text).strip()
            text = text[:5000]  # Limit to 5000 chars
        
        # ============================================
        # Try to get publish date
        # ============================================
        publish_date = ""
        date_selectors = [
            "meta[property='article:published_time']",
            "meta[name='date']",
            "time[datetime]",
            ".publish-date",
            ".date"
        ]
        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                if elem.get("content"):
                    publish_date = elem["content"]
                elif elem.get("datetime"):
                    publish_date = elem["datetime"]
                elif elem.get_text(strip=True):
                    publish_date = elem.get_text(strip=True)
                if publish_date:
                    break
        
        return {
            "title": title or "",
            "text": text or "",
            "authors": [],
            "top_image": "",
            "publish_date": publish_date,
        }
        
    except Exception as e:
        logger.error(f"BS4 extraction failed: {e}")
        return {
            "title": "",
            "text": "",
            "authors": [],
            "top_image": "",
            "publish_date": "",
        }


class URLScraper:
    """
    Extract article title and content from news URLs
    Uses 3-tier fallback system:
    1. cloudscraper (Cloudflare bypass)
    2. Playwright (headless browser)
    3. Wayback Machine (cached copy)
    """

    def __init__(self):
        self.session = None
        self._init_session()
        logger.info("🌐 URLScraper initialized")

    def _init_session(self):
        """Initialize session with cloudscraper if available"""
        if CLOUDSCRAPER_AVAILABLE:
            self.session = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            logger.info("   Using cloudscraper (Cloudflare bypass)")
        else:
            self.session = requests.Session()
            logger.info("   Using requests (install cloudscraper for better results)")

    def scrape_url(self, url: str) -> dict:
        """
        Main scraping method with 3-tier fallback
        
        Returns:
            dict with keys: success, source, data, error
        """
        html = None
        tier = None

        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        logger.info(f"🌍 Scraping URL: {url}")

        # ============================================
        # TIER 1: cloudscraper with realistic headers
        # ============================================
        try:
            time.sleep(random.uniform(1.0, 2.5))  # Polite delay
            headers = _build_headers(url)
            response = self.session.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Check if it's a blocking page (Cloudflare)
                if "Just a moment" not in response.text[:500] and "403" not in response.text[:100]:
                    html = response.text
                    tier = "cloudscraper"
                    logger.info(f"✅ TIER 1: cloudscraper succeeded")
        except Exception as e:
            logger.warning(f"TIER 1 (cloudscraper) failed: {e}")

        # ============================================
        # TIER 2: Playwright headless browser
        # ============================================
        if not html:
            logger.info("TIER 1 failed, trying TIER 2: Playwright...")
            html = _fetch_with_playwright(url)
            if html:
                tier = "playwright"
                logger.info(f"✅ TIER 2: Playwright succeeded")

        # ============================================
        # TIER 3: Wayback Machine cached copy
        # ============================================
        if not html:
            logger.info("TIER 2 failed, trying TIER 3: Wayback Machine...")
            html = _fetch_from_archive(url)
            if html:
                tier = "archive.org"
                logger.info(f"✅ TIER 3: Wayback Machine succeeded")

        # ============================================
        # If all tiers failed
        # ============================================
        if not html:
            logger.error(f"❌ All fetch tiers failed for URL: {url}")
            return {
                "success": False,
                "source": "none",
                "data": {},
                "error": "All fetch tiers failed - site may be blocking requests"
            }

        # ============================================
        # Extract article from HTML
        # ============================================
        data = _extract_article(html, url)
        
        if data.get("title"):
            logger.info(f"✅ Extracted: {data['title'][:80]}...")
        else:
            logger.warning("⚠️ Extracted article has no title")

        return {
            "success": True,
            "source": tier,
            "data": data,
            "error": ""
        }

    # ============================================
    # Legacy method for compatibility with orchestrator
    # ============================================

    def scrape_article(self, url: str) -> dict:
        """
        Wrapper method for orchestrator compatibility
        Returns dict with headline, text, domain, etc.
        """
        result = self.scrape_url(url)
        
        if result.get("success"):
            data = result.get("data", {})
            return {
                "success": True,
                "headline": data.get("title", ""),
                "text": data.get("text", ""),
                "claim_text": data.get("title", "") + " " + (data.get("text", "")[:500]),
                "domain": urlparse(url).netloc,
                "url": url,
                "authors": data.get("authors", []),
                "publish_date": data.get("publish_date", ""),
                "scrape_source": result.get("source", "unknown")
            }
        else:
            return {
                "success": False,
                "error": result.get("error", "Scraping failed"),
                "headline": "",
                "text": "",
                "domain": urlparse(url).netloc,
                "url": url
            }

    # ============================================
    # File download detection (kept from original)
    # ============================================

    def check_for_downloads(self, url: str) -> dict:
        """Check if URL points to a downloadable file"""
        result = {
            'is_download': False,
            'data': None,
            'file_path': None,
            'filename': None,
            'suffix': '',
            'content_type': None,
            'size': 0,
            'success': False
        }
        
        try:
            file_extensions = {
                '.exe': '.exe', '.dll': '.dll', '.msi': '.msi',
                '.pdf': '.pdf', '.doc': '.doc', '.docx': '.docx',
                '.zip': '.zip', '.rar': '.rar', '.7z': '.7z',
                '.txt': '.txt', '.bat': '.bat', '.ps1': '.ps1',
                '.vbs': '.vbs', '.js': '.js', '.jar': '.jar',
                '.apk': '.apk', '.deb': '.deb', '.rpm': '.rpm',
                '.iso': '.iso', '.img': '.img', '.bin': '.bin'
            }
            
            parsed = urlparse(url)
            path = parsed.path.lower()
            
            # Check by file extension
            for ext, suffix in file_extensions.items():
                if path.endswith(ext):
                    result['is_download'] = True
                    result['suffix'] = suffix
                    filename = path.split('/')[-1]
                    result['filename'] = filename or f"download{suffix}"
                    logger.info(f"📥 Download detected by extension: {ext}")
                    break
            
            # Also check content-type
            if not result['is_download']:
                try:
                    response = self.session.head(url, timeout=5, allow_redirects=True)
                    content_type = response.headers.get('Content-Type', '').lower()
                    
                    binary_types = [
                        'application/octet-stream', 'application/x-msdownload',
                        'application/exe', 'application/pdf', 'application/zip',
                        'application/x-rar-compressed', 'application/x-7z-compressed'
                    ]
                    
                    for bt in binary_types:
                        if bt in content_type:
                            result['is_download'] = True
                            result['content_type'] = content_type
                            logger.info(f"📥 Download detected by content-type: {content_type}")
                            break
                except Exception as e:
                    logger.debug(f"HEAD request failed: {e}")
            
            # Download if it's a file
            if result['is_download']:
                response = self.session.get(url, timeout=30, stream=True)
                if response.status_code == 200:
                    content = b''
                    total_size = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                        total_size += len(chunk)
                        if total_size > 50 * 1024 * 1024:
                            logger.warning(f"⚠️ File too large (>50MB), truncating")
                            break
                    
                    result['data'] = content
                    result['size'] = total_size
                    result['success'] = True
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(
                        delete=False, 
                        suffix=result['suffix'] or '.bin',
                        prefix='verify_ai_'
                    ) as tmp_file:
                        tmp_file.write(content)
                        result['file_path'] = tmp_file.name
                    
                    logger.info(f"✅ Downloaded {total_size} bytes to {result['file_path']}")
                    
        except Exception as e:
            logger.error(f"❌ Download check failed: {e}")
            result['error'] = str(e)
        
        return result