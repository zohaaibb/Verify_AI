# app/services/image_analysis_pipeline.py
"""
TruthLens Image Analysis Pipeline
- OCR text extraction with preprocessing
- Smart headline/body separation using layout analysis
- Digital forensics (EXIF, metadata)
- Hidden payload detection (improved PE validation, trailing data, URL detection)
- Deepfake detection (with confidence threshold 0.5)
- Security hardening: magic byte validation, SVG sanitisation
- Lightweight steganalysis (chi‑square + RS, no ML model)
"""
import logging
import os
import re
import hashlib
import struct
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from PIL import Image, ImageEnhance
import pytesseract
import exifread
import cv2
import numpy as np

from app.services.deepfake_detector import DeepfakeAnalyzer

logger = logging.getLogger(__name__)

# Tesseract configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'

try:
    tesseract_version = pytesseract.get_tesseract_version()
    logger.info(f"✅ Tesseract {tesseract_version} initialized")
except Exception as e:
    logger.warning(f"⚠️ Tesseract not found: {e}")

# ── Forensics constants (new) ────────────────────────────────────────────────
JPEG_EOI = b'\xff\xd9'
PNG_IEND = b'IEND\xaeB`\x82'

_DOUBLE_EXT_RE = re.compile(
    r'\.(jpe?g|png|gif|bmp|webp|tiff?)\.[a-zA-Z0-9]{1,10}$',
    re.IGNORECASE,
)

_URL_RE = re.compile(
    rb'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]{8,}',
    re.IGNORECASE,
)

_CHI_SQ_THRESHOLD = 0.95


def _regularised_gamma_lower(a: float, x: float) -> float:
    """Regularised lower incomplete gamma P(a, x) via series expansion."""
    import math
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    MAX_ITER = 200
    TOL = 1e-10
    term = 1.0 / a
    total = term
    for n in range(1, MAX_ITER):
        term *= x / (a + n)
        total += term
        if abs(term) < TOL:
            break
    try:
        log_prefix = -x + a * math.log(x) - math.lgamma(a)
        return math.exp(log_prefix) * total
    except (ValueError, OverflowError):
        return 0.0


@dataclass
class TextBlock:
    """Represents a block of text from OCR with layout metadata."""
    text: str
    font_size: float
    word_count: int
    line_count: int
    top_ratio: float
    block_type: str = ""
    confidence: float = 0.0


class ImageAnalysisPipeline:
    """
    Complete image analysis pipeline integrating:
    - Smart OCR with headline/body separation
    - Digital forensics (EXIF, metadata)
    - Hidden payload detection (validates PE structure, trailing data, URL detection)
    - Deepfake detection (thresholded at 0.5 confidence)
    - Security hardening: magic byte validation, SVG sanitisation
    - Lightweight steganalysis (chi‑square + RS, no ML model)
    """

    MAGIC_BYTES = {
        b'\xff\xd8\xff': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'RIFF': 'WEBP',
        b'BM': 'BMP',
        b'\x00\x00\x01\x00': 'ICO',
    }

    DEEPFAKE_CONFIDENCE_THRESHOLD = 0.5

    def __init__(self):
        self.ocr = pytesseract
        self.corrections = {
            'lunches': 'launches', 'Madel': 'Model', 'Vin': 'in',
            'PTl': 'PTI', 'prote tin': 'protest in', 'protetin': 'protest in',
            'Paja': 'Raja', 'day': 'today', 'announces rotestin': 'announces protest in',
            'loses85': 'loses 85', 'visiondoctors': 'vision, doctors',
            'confirmPTannounces': 'confirm\nPTI announces', 'loses85%': 'loses 85%',
            '85%vision': '85% vision', 'ImranKhan': 'Imran Khan', 'SalmanAkram': 'Salman Akram',
            'Rojas': 'Raja', 'lslamabad': 'Islamabad', 'PMLN': 'PML-N', 'PPP': 'PPP',
            'PTI': 'PTI', 'winscricket': 'wins cricket', 'avers': 'overs',
            'against india': 'against India', 'winscricket match': 'wins cricket match',
        }
        self.deepfake_analyzer = None
        try:
            self.deepfake_analyzer = DeepfakeAnalyzer(detector_type="ensemble")
            if self.deepfake_analyzer and hasattr(self.deepfake_analyzer, 'get_info'):
                info = self.deepfake_analyzer.get_info()
                logger.info(f"✅ Deepfake detector initialized: {info['detector']['name']}")
            else:
                self.deepfake_analyzer = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize deepfake detector: {e}")
            self.deepfake_analyzer = None
        logger.info("✅ Image Analysis Pipeline initialized")

    # ================================================================== #
    # Security hardening – magic byte validation & SVG sanitisation
    # ================================================================== #
    def validate_image_magic_bytes(self, image_path: str) -> dict:
        try:
            with open(image_path, 'rb') as f:
                header = f.read(16)
            detected = None
            for magic, fmt in self.MAGIC_BYTES.items():
                if header.startswith(magic):
                    detected = fmt
                    if fmt == 'WEBP' and header[8:12] != b'WEBP':
                        detected = 'RIFF (non-WEBP)'
                    break
            if detected is None:
                return {'valid': False, 'detected_format': 'UNKNOWN',
                        'warning': 'File magic bytes do not match any known image format'}
            ext = image_path.rsplit('.', 1)[-1].lower() if '.' in image_path else ''
            ext_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG', 'gif': 'GIF', 'webp': 'WEBP', 'bmp': 'BMP'}
            expected = ext_map.get(ext, '')
            if expected and detected != expected:
                return {'valid': False, 'detected_format': detected,
                        'warning': f"Extension .{ext} but file is {detected} — possible spoofing"}
            return {'valid': True, 'detected_format': detected, 'warning': None}
        except Exception as e:
            return {'valid': False, 'detected_format': 'ERROR', 'warning': str(e)}

    def sanitise_svg(self, svg_path: str) -> dict:
        try:
            with open(svg_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return {'safe': False, 'sanitised': False, 'threats_removed': [str(e)]}
        threats = []
        if re.search(r'<script[\s>]', content, re.I):
            threats.append('embedded <script> tag')
        if re.search(r'(href|src|xlink:href)\s*=\s*["\']?\s*javascript:', content, re.I):
            threats.append('javascript: URI in attribute')
        if re.search(r'\bon\w+\s*=', content, re.I):
            threats.append('inline event handler')
        if re.search(r'<foreignObject', content, re.I):
            threats.append('<foreignObject> (potential HTML injection)')
        if not threats:
            return {'safe': True, 'sanitised': False, 'threats_removed': []}
        sanitised = content
        sanitised = re.sub(r'<script[\s\S]*?</script>', '', sanitised, flags=re.I)
        sanitised = re.sub(r'javascript:[^"\'\s>]*', '#', sanitised, flags=re.I)
        sanitised = re.sub(r'\bon\w+\s*=\s*["\'][^"\']*["\']', '', sanitised, flags=re.I)
        sanitised = re.sub(r'<foreignObject[\s\S]*?</foreignObject>', '', sanitised, flags=re.I)
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(sanitised)
        return {'safe': False, 'sanitised': True, 'threats_removed': threats}

    # ================================================================== #
    # SMART TEXT EXTRACTION (unchanged)
    # ================================================================== #
    def extract_structured_text(self, image_path: str) -> Dict[str, Any]:
        result = {
            'headline': '',
            'body': '',
            'suggested_text': '',
            'full_ocr_text': '',
            'method': 'structured_blocks',
            'extraction_confidence': 0.0,
            'warning': ''
        }
        try:
            img = self._load_and_preprocess(image_path)
            img_h = img.height if img.height > 0 else 1
            data = pytesseract.image_to_data(img, config="--oem 3 --psm 3", output_type=pytesseract.Output.DICT)
            blocks = {}
            n = len(data["text"])
            for i in range(n):
                word = data["text"][i].strip()
                conf = int(data["conf"][i])
                if not word or conf < 20:
                    continue
                bnum = data["block_num"][i]
                if bnum not in blocks:
                    blocks[bnum] = []
                blocks[bnum].append({"word": word, "height": data["height"][i], "top": data["top"][i], "conf": conf})
            if not blocks:
                return self._extract_flat_fallback(img, result)
            text_blocks = []
            for bnum, words in blocks.items():
                block_text = " ".join(w["word"] for w in words)
                block_text = re.sub(r"[ \t]+", " ", block_text).strip()
                if not block_text:
                    continue
                avg_height = sum(w["height"] for w in words) / len(words)
                avg_top = sum(w["top"] for w in words) / len(words)
                top_ratio = avg_top / img_h
                word_count = len(block_text.split())
                tb = TextBlock(
                    text=block_text, font_size=avg_height, word_count=word_count,
                    line_count=1, top_ratio=top_ratio,
                    confidence=sum(w["conf"] for w in words) / len(words) / 100.0
                )
                text_blocks.append(tb)
            if not text_blocks:
                return self._extract_flat_fallback(img, result)
            max_font = max(tb.font_size for tb in text_blocks)
            for tb in text_blocks:
                if tb.top_ratio < 0.08 or tb.top_ratio > 0.88:
                    tb.block_type = "noise"; continue
                if tb.word_count <= 3:
                    tb.block_type = "noise"; continue
                font_ratio = tb.font_size / max_font if max_font > 0 else 0
                if font_ratio >= 0.70 and tb.word_count <= 20:
                    tb.block_type = "headline"; continue
                if tb.word_count >= 8:
                    tb.block_type = "body"; continue
                tb.block_type = "noise"
            result['full_ocr_text'] = "\n\n".join(tb.text for tb in text_blocks)
            headlines = [tb for tb in text_blocks if tb.block_type == "headline"]
            bodies = [tb for tb in text_blocks if tb.block_type == "body"]
            if headlines:
                result['headline'] = headlines[0].text
            if bodies:
                result['body'] = "\n\n".join(tb.text for tb in bodies[:2])
            result['suggested_text'] = self._assemble_suggested(result['headline'], result['body'])
            conf_score = 0.0
            if headlines:
                conf_score += 0.40 + min(headlines[0].confidence * 0.20, 0.20)
            if bodies:
                conf_score += 0.30 + min(bodies[0].confidence * 0.10, 0.10)
            result['extraction_confidence'] = min(round(conf_score, 2), 1.0)
            if not result['suggested_text']:
                return self._extract_flat_fallback(img, result)
        except Exception as e:
            logger.error(f"Smart extraction failed: {e}")
            result['warning'] = str(e)
            result['method'] = 'error'
        return result

    def _extract_flat_fallback(self, img: Image.Image, result: Dict) -> Dict:
        result['method'] = 'flat_fallback'
        try:
            raw = pytesseract.image_to_string(img, config="--oem 3 --psm 6")
            result['full_ocr_text'] = raw
            lines = [l.strip() for l in raw.splitlines() if l.strip() and len(l.split()) >= 4]
            if lines:
                result['suggested_text'] = " ".join(lines[:3])
            result['extraction_confidence'] = 0.55
        except Exception as e:
            result['warning'] = str(e)
        return result

    def _assemble_suggested(self, headline: str, body: str) -> str:
        parts = []
        if headline:
            parts.append(headline.strip())
        if body:
            sentences = re.split(r'(?<=[.!?])\s+', body.strip())
            parts.append(" ".join(sentences[:3]))
        return "\n\n".join(parts)

    def _load_and_preprocess(self, image_path: str) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) < 1200:
            scale = 1200 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.3)
        return img

    # ================================================================== #
    # REGULAR OCR EXTRACTION (unchanged)
    # ================================================================== #
    def extract_text_from_image(self, image_path: str) -> Dict:
        result = {'text': '', 'method': 'none', 'confidence': 0, 'word_count': 0, 'quality': 'unknown'}
        try:
            img = Image.open(image_path)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            best_text = ""
            best_method = "standard"
            max_words = 0
            for method_name, processed in [
                ("standard", img),
                ("contrast", ImageEnhance.Contrast(img).enhance(2.0)),
                ("sharpness", ImageEnhance.Sharpness(img).enhance(2.0)),
            ]:
                text = self.ocr.image_to_string(processed)
                words = len(text.split())
                if words > max_words:
                    max_words = words
                    best_text = text
                    best_method = method_name
            text4 = self.ocr.image_to_string(img, config='--oem 3 --psm 6')
            words4 = len(text4.split())
            if words4 > max_words:
                max_words = words4
                best_text = text4
                best_method = "custom_config"
            for name, opencv_img in self._preprocess_image_cv(img):
                text = self.ocr.image_to_string(Image.fromarray(opencv_img))
                words = len(text.split())
                if words > max_words:
                    max_words = words
                    best_text = text
                    best_method = name
            cleaned = self._clean_text(best_text)
            quality = 'good' if max_words >= 8 else 'fair' if max_words >= 4 else 'poor' if max_words > 0 else 'none'
            confidence = 0.8 if max_words >= 8 else 0.5 if max_words >= 4 else 0.3 if max_words > 0 else 0
            result = {'text': cleaned, 'method': best_method, 'confidence': confidence,
                      'word_count': max_words, 'quality': quality, 'raw_text': best_text[:200] if best_text else ''}
            logger.info(f"📝 OCR {quality}: {max_words} words via {best_method}")
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            result['error'] = str(e)
        return result

    def _preprocess_image_cv(self, image):
        versions = []
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            versions.append(('grayscale', gray))
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            versions.append(('threshold', thresh))
            denoised = cv2.medianBlur(gray, 3)
            versions.append(('denoised', denoised))
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            versions.append(('adaptive', adaptive))
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            versions.append(('sharpened', sharpened))
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
        return versions

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\d+)%', r'\1%', text)
        text = re.sub(r'(\d+) (\d+)', r'\1\2', text)
        for wrong, correct in self.corrections.items():
            text = text.replace(wrong, correct)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        return text.strip()

    # ================================================================== #
    # DEEPFAKE DETECTION (unchanged)
    # ================================================================== #
    def detect_deepfake(self, image_path: str) -> Dict[str, Any]:
        if self.deepfake_analyzer is None:
            return {
                'success': False,
                'error': 'Deepfake analyzer not initialized',
                'deepfake_result': {
                    'is_deepfake': False, 'confidence': 0.0, 'model_name': 'none',
                }
            }
        return self.deepfake_analyzer.analyze(image_path)

    # ================================================================== #
    # COMPLETE IMAGE ANALYSIS (with security checks integrated)
    # ================================================================== #
    def analyze_image(self, image_path: str, include_deepfake: bool = True) -> Dict:
        results = {
            'success': False,
            'image_info': {},
            'ocr_results': {},
            'forensics': {},
            'hidden_payload': {},
            'deepfake_results': {},
            'image_hash': '',
            'analysis_id': '',
            'image_security': {},
        }
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            magic = self.validate_image_magic_bytes(image_path)
            results['image_security']['magic_bytes'] = magic
            if not magic['valid']:
                results['hidden_payload']['warnings'].append(magic['warning'])
                results['hidden_payload']['has_payload'] = True
            if image_path.lower().endswith('.svg'):
                svg_check = self.sanitise_svg(image_path)
                results['image_security']['svg'] = svg_check
                if not svg_check['safe']:
                    for threat in svg_check['threats_removed']:
                        results['hidden_payload']['warnings'].append(f"SVG threat: {threat}")
                    results['hidden_payload']['has_payload'] = True
            img = Image.open(image_path)
            file_size = os.path.getsize(image_path)
            with open(image_path, 'rb') as f:
                image_hash = hashlib.md5(f.read()).hexdigest()
            results['image_info'] = {
                'format': img.format, 'width': img.width, 'height': img.height,
                'mode': img.mode, 'file_size': file_size, 'file_name': os.path.basename(image_path)
            }
            results['image_hash'] = image_hash
            results['analysis_id'] = hashlib.md5(f"{image_hash}_{datetime.now()}".encode()).hexdigest()[:8]
            logger.info(f"📸 Image: {img.format} {img.width}x{img.height} ({file_size} bytes)")
            ocr_results = self.extract_text_from_image(image_path)
            results['ocr_results'] = ocr_results
            results['forensics'] = self._run_forensics(image_path)
            results['hidden_payload'] = self._detect_hidden_payload(image_path, img)
            if include_deepfake:
                deepfake_result = self.detect_deepfake(image_path)
                if deepfake_result.get('success'):
                    df = deepfake_result.get('deepfake_result', {})
                    confidence = df.get('confidence', 0.0)
                    if confidence < self.DEEPFAKE_CONFIDENCE_THRESHOLD:
                        df['is_deepfake'] = False
                        df['verdict'] = 'Authentic'
                        logger.info(f"Deepfake confidence {confidence:.2%} below threshold – treating as Authentic")
                    else:
                        if df.get('is_deepfake'):
                            logger.warning(f"🚨 DEEPFAKE DETECTED! Confidence: {confidence:.2%}")
                results['deepfake_results'] = deepfake_result
            results['steganalysis'] = self._steganalysis(image_path)
            results['success'] = True
        except Exception as e:
            logger.error(f"❌ Image analysis failed: {e}")
            results['error'] = str(e)
        return results

    def _run_forensics(self, image_path):
        forensics = {'exif_data': {}, 'has_gps': False, 'edit_history': [], 'warnings': [], 'timestamp': None}
        try:
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)
            for tag, value in tags.items():
                try:
                    forensics['exif_data'][tag] = str(value)[:200]
                except:
                    pass
                if 'GPS' in tag:
                    forensics['has_gps'] = True
                if 'Software' in tag:
                    forensics['edit_history'].append(f"Edited with: {value}")
                if 'DateTimeOriginal' in tag:
                    forensics['timestamp'] = str(value)
        except Exception as e:
            logger.warning(f"Forensics extraction failed: {e}")
        return forensics

    # ================================================================== #
    # IMPROVED HIDDEN PAYLOAD DETECTION (fully implemented)
    # ================================================================== #
    def _detect_hidden_payload(self, image_path, img):
        result = {
            'has_payload': False, 'warnings': [], 'suspicious_indicators': [],
            'payload_size': 0, 'trailing_urls': [],
        }
        try:
            file_size = os.path.getsize(image_path)
            with open(image_path, 'rb') as f:
                file_data = f.read()
            trailing_bytes = None
            if file_data[:2] == b'\xff\xd8':
                eoi_pos = file_data.rfind(JPEG_EOI)
                if eoi_pos != -1:
                    after_eoi = eoi_pos + len(JPEG_EOI)
                    if after_eoi < file_size:
                        trailing_bytes = file_data[after_eoi:]
                        n = len(trailing_bytes)
                        result['warnings'].append(f'🔴 {n:,} bytes after JPEG EOI marker')
                        result['has_payload'] = True
                        result['suspicious_indicators'].append('jpeg_trailing_data')
                        result['payload_size'] = max(result['payload_size'], n)
            elif file_data[:8] == b'\x89PNG\r\n\x1a\n':
                iend_pos = file_data.rfind(PNG_IEND)
                if iend_pos != -1:
                    after_iend = iend_pos + len(PNG_IEND)
                    if after_iend < file_size:
                        trailing_bytes = file_data[after_iend:]
                        n = len(trailing_bytes)
                        result['warnings'].append(f'🔴 {n:,} bytes after PNG IEND chunk')
                        result['has_payload'] = True
                        result['suspicious_indicators'].append('png_trailing_data')
                        result['payload_size'] = max(result['payload_size'], n)
            scan_region = trailing_bytes if trailing_bytes is not None else file_data[-4096:]
            found_urls = [m.group(0).decode('ascii', errors='replace') for m in _URL_RE.finditer(scan_region)]
            if found_urls:
                result['trailing_urls'] = found_urls
                result['has_payload'] = True
                result['suspicious_indicators'].append('appended_urls')
                for url in found_urls:
                    result['warnings'].append(f'🌐 Appended URL found: {url}')
            try:
                img_data_size = len(img.tobytes())
                overhead_threshold = 5000
                if file_size > img_data_size + overhead_threshold:
                    diff = file_size - (img_data_size + overhead_threshold)
                    result['payload_size'] = max(result['payload_size'], diff)
                    result['warnings'].append(f'⚠️ File has {diff:,} bytes of suspicious appended data')
                    result['has_payload'] = True
                    result['suspicious_indicators'].append('appended_data_heuristic')
            except Exception as e:
                logger.debug(f"Size heuristic failed: {e}")
            for sig in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
                if sig in file_data:
                    offset = file_data.find(sig)
                    result['warnings'].append(f'📦 ZIP archive signature at offset {offset:,}')
                    result['has_payload'] = True
                    result['suspicious_indicators'].append('embedded_zip')
            mz_offset = file_data.find(b'MZ')
            if mz_offset != -1:
                if mz_offset + 0x40 <= file_size:
                    pe_offset = int.from_bytes(file_data[mz_offset+0x3C:mz_offset+0x40], 'little')
                    if mz_offset + pe_offset + 4 <= file_size:
                        pe_sig = file_data[mz_offset + pe_offset : mz_offset + pe_offset + 4]
                        if pe_sig == b'PE\0\0':
                            result['warnings'].append(f'⚙️ Valid PE executable detected at offset {mz_offset:,}')
                            result['has_payload'] = True
                            result['suspicious_indicators'].append('hidden_exe')
            if b'%PDF' in file_data:
                offset = file_data.find(b'%PDF')
                result['warnings'].append(f'📄 PDF signature at offset {offset:,}')
                result['has_payload'] = True
                result['suspicious_indicators'].append('embedded_pdf')
            iend_count = file_data.count(b'IEND')
            if iend_count > 1:
                result['warnings'].append(f'🔴 Multiple IEND markers ({iend_count})')
                result['has_payload'] = True
                result['suspicious_indicators'].append('multiple_iend')
            soi_count = file_data.count(b'\xff\xd8')
            if soi_count > 1:
                result['warnings'].append(f'🖼️ Multiple JPEG SOI markers ({soi_count})')
                result['has_payload'] = True
                result['suspicious_indicators'].append('multiple_soi')
            if _DOUBLE_EXT_RE.search(image_path):
                result['warnings'].append('⚠️ Double extension detected – file may be masquerading')
                result['has_payload'] = True
                result['suspicious_indicators'].append('double_extension')
        except Exception as e:
            logger.warning(f"Hidden payload detection failed: {e}")
        result['suspicious_indicators'] = list(dict.fromkeys(result['suspicious_indicators']))
        if result['has_payload']:
            logger.warning(f"🚨 HIDDEN PAYLOAD DETECTED: {result['suspicious_indicators']}")
        return result

    # ================================================================== #
    # NEW: Steganalysis (chi‑square + RS)
    # ================================================================== #
    def _steganalysis(self, image_path: str) -> Dict[str, Any]:
        result = {'chi_square': None, 'rs_analysis': None,
                  'steg_suspected': False, 'confidence': 'none', 'warnings': []}
        try:
            img = Image.open(image_path).convert('RGB')
            arr = np.array(img, dtype=np.uint8)
        except Exception as e:
            result['error'] = str(e)
            return result
        chi_results = self._chi_square_attack(arr)
        result['chi_square'] = chi_results
        rs_results = self._rs_analysis(arr)
        result['rs_analysis'] = rs_results
        chi_flag = chi_results.get('suspicious', False)
        rs_flag = rs_results.get('suspicious', False)
        if chi_flag and rs_flag:
            result['steg_suspected'] = True
            result['confidence'] = 'high'
            result['warnings'].append('🕵️ Both chi-square AND RS analysis flag potential steganography')
        elif chi_flag or rs_flag:
            result['steg_suspected'] = True
            result['confidence'] = 'medium'
            method = 'chi-square' if chi_flag else 'RS analysis'
            result['warnings'].append(f'🕵️ {method} flags potential steganography (single-method signal)')
        if result['steg_suspected']:
            logger.warning(f"🚨 Steg suspected (confidence={result['confidence']}): {image_path}")
        return result

    def _chi_square_attack(self, arr) -> Dict[str, Any]:
        import math
        channel_names = ['R', 'G', 'B']
        channel_results = {}
        suspicious_count = 0
        for c, name in enumerate(channel_names):
            channel = arr[:, :, c].flatten()
            freq = np.bincount(channel, minlength=256).astype(float)
            expected = np.zeros(256)
            for k in range(0, 256, 2):
                pair_total = freq[k] + freq[k + 1]
                expected[k] = expected[k + 1] = pair_total / 2.0
            mask = expected > 0
            chi_sq = float(np.sum(((freq[mask] - expected[mask]) ** 2) / expected[mask]))
            df = 127
            try:
                p_value = 1.0 - _regularised_gamma_lower(df / 2, chi_sq / 2)
            except Exception:
                p_value = 0.0
            suspicious = p_value > 0.95
            if suspicious:
                suspicious_count += 1
            channel_results[name] = {'chi_sq': round(chi_sq, 2), 'p_value': round(p_value, 4), 'suspicious': suspicious}
        return {'channels': channel_results, 'suspicious': suspicious_count >= 2,
                'note': 'p_value close to 1.0 = LSBs suspiciously uniform = possible steg'}

    def _rs_analysis(self, arr) -> Dict[str, Any]:
        channel_names = ['R', 'G', 'B']
        channel_results = {}
        suspicious_count = 0
        for c, name in enumerate(channel_names):
            channel = arr[:, :, c]
            H, W = channel.shape
            W4 = (W // 4) * 4
            if W4 == 0:
                channel_results[name] = {'error': 'image too narrow'}
                continue
            pixels = channel[:, :W4].reshape(-1, 4).astype(np.int16)
            def discriminate(p):
                return (np.abs(p[:, 0] - p[:, 1]) + np.abs(p[:, 1] - p[:, 2]) + np.abs(p[:, 2] - p[:, 3]))
            def flip_lsb(p):
                return p ^ 1
            f_orig = discriminate(pixels)
            f_flip = discriminate(flip_lsb(pixels))
            R = np.sum(f_flip > f_orig)
            S = np.sum(f_flip < f_orig)
            RS_total = R + S
            if RS_total == 0:
                channel_results[name] = {'error': 'degenerate (flat image?)'}
                continue
            rs_ratio = abs(R - S) / RS_total
            suspicious = rs_ratio < 0.10
            if suspicious:
                suspicious_count += 1
            channel_results[name] = {'R': int(R), 'S': int(S), 'rs_ratio': round(float(rs_ratio), 4),
                                     'suspicious': suspicious,
                                     'note': 'rs_ratio near 0 = R≈S = possible steg; near 1 = clean'}
        return {'channels': channel_results, 'suspicious': suspicious_count >= 2}

    def get_text_for_analysis(self, image_path: str) -> str:
        ocr_result = self.extract_text_from_image(image_path)
        return ocr_result.get('text', '')