
import os
import re
import hashlib
import logging
import struct
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

JPEG_EOI = b'\xff\xd9'          # JPEG End-Of-Image marker
PNG_IEND = b'IEND\xaeB`\x82'   # full 12-byte PNG IEND chunk (type + CRC)

# Stronger double-extension: catches .jpg.exe, .png.php, .gif.js, etc.
# Matches any file whose name has a known-image extension followed by
# ANY non-empty extension (the dangerous outer extension).
_DOUBLE_EXT_RE = re.compile(
    r'\.(jpe?g|png|gif|bmp|webp|tiff?)\.[a-zA-Z0-9]{1,10}$',
    re.IGNORECASE,
)

# URL pattern for trailing-data URL scan
_URL_RE = re.compile(
    rb'https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]{8,}',
    re.IGNORECASE,
)

# Chi-square significance threshold.  LSB chi-square values above this
# strongly suggest the LSBs are uniform (hidden data).  0.95 → p < 0.05.
_CHI_SQ_THRESHOLD = 0.95        # fraction of channels exceeding chi-sq critical


class ImageForensics:
    """Digital forensics for images and files."""

    def __init__(self):
        logger.info("🔬 Image Forensics module initialized")

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """
        Perform forensic analysis on a file.
        Works for images and executables.
        """
        results = {
            'file_name':      os.path.basename(file_path),
            'file_size':      os.path.getsize(file_path),
            'file_hash':      self._calculate_hashes(file_path),
            'file_type':      self._detect_file_type(file_path),
            'hidden_payload': self._detect_hidden_payload(file_path),
        }

        if results['file_type'].startswith('image/'):
            results['image_analysis'] = self._analyze_image(file_path)
            results['steganalysis']   = self._steganalysis(file_path)

        return results

    # ── Unchanged helpers ──────────────────────────────────────────────────────

    def _calculate_hashes(self, file_path: str) -> Dict[str, str]:
        hashes = {}
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            hashes['md5']    = hashlib.md5(data).hexdigest()
            hashes['sha256'] = hashlib.sha256(data).hexdigest()
        except Exception as e:
            hashes['error'] = str(e)
        return hashes

    def _detect_file_type(self, file_path: str) -> str:
        try:
            import magic
            return magic.from_file(file_path, mime=True)
        except Exception:
            ext = os.path.splitext(file_path)[1].lower()
            return {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png',  '.gif': 'image/gif',
            }.get(ext, 'unknown')

    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        analysis = {
            'metadata':   self._extract_metadata(image_path),
            'dimensions': None,
            'mode':       None,
            'format':     None,
        }
        try:
            img = Image.open(image_path)
            analysis['dimensions'] = img.size
            analysis['mode']       = img.mode
            analysis['format']     = img.format
        except Exception as e:
            analysis['error'] = str(e)
        return analysis

    def _extract_metadata(self, image_path: str) -> Dict[str, Any]:
        metadata = {
            'has_exif': False,
            'data':     {},
            'camera':   None,
            'datetime': None,
            'gps':      False,
        }
        try:
            img = Image.open(image_path)
            if hasattr(img, '_getexif') and img._getexif():
                exif = img._getexif()
                metadata['has_exif'] = True
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    metadata['data'][str(tag)] = str(value)
                metadata['camera']   = f"{metadata['data'].get('Make', '')} {metadata['data'].get('Model', '')}".strip()
                metadata['datetime'] = metadata['data'].get('DateTime')
                metadata['gps']      = 'GPSInfo' in metadata['data']
        except Exception as e:
            metadata['error'] = str(e)
        return metadata

    # ── UPDATED: _detect_hidden_payload ───────────────────────────────────────

    def _detect_hidden_payload(self, file_path: str) -> Dict[str, Any]:
        """
        Detect hidden payload in a file.

        Improvements over the original:
          1. Generic JPEG / PNG trailing-data detection via formal EOF markers.
          2. Stronger double-extension regex (catches .jpg.exe, .png.php, etc.).
          3. Appended URL detection in trailing bytes.
          4. All original checks (ZIP/PE/PDF signatures, multiple IEND/SOI,
             appended-size heuristic) are preserved.
        """
        result = {
            'has_payload':           False,
            'warnings':              [],
            'suspicious_indicators': [],
            'payload_size':          0,
            'trailing_urls':         [],
        }

        try:
            with open(file_path, 'rb') as f:
                file_data = f.read()

            file_size = len(file_data)

            # ── 1. Generic trailing-data after formal EOF marker ──────────────

            trailing_bytes: Optional[bytes] = None

            if file_data[:2] == b'\xff\xd8':            # JPEG magic
                eoi_pos = file_data.rfind(JPEG_EOI)
                if eoi_pos != -1:
                    after_eoi = eoi_pos + len(JPEG_EOI)
                    if after_eoi < file_size:
                        trailing_bytes = file_data[after_eoi:]
                        n = len(trailing_bytes)
                        result['warnings'].append(
                            f'🔴 {n:,} bytes after JPEG EOI (0xFF 0xD9) marker'
                        )
                        result['has_payload'] = True
                        result['suspicious_indicators'].append('jpeg_trailing_data')
                        result['payload_size'] = max(result['payload_size'], n)
                        logger.warning(f"🚨 JPEG trailing data: {n:,} bytes")

            elif file_data[:8] == b'\x89PNG\r\n\x1a\n':  # PNG magic
                iend_pos = file_data.rfind(PNG_IEND)
                if iend_pos != -1:
                    after_iend = iend_pos + len(PNG_IEND)
                    if after_iend < file_size:
                        trailing_bytes = file_data[after_iend:]
                        n = len(trailing_bytes)
                        result['warnings'].append(
                            f'🔴 {n:,} bytes after PNG IEND chunk'
                        )
                        result['has_payload'] = True
                        result['suspicious_indicators'].append('png_trailing_data')
                        result['payload_size'] = max(result['payload_size'], n)
                        logger.warning(f"🚨 PNG trailing data: {n:,} bytes")

            # ── 2. Appended URL detection (scan trailing region or full file) ──

            # Prefer scanning only the trailing region for speed; if no formal
            # trailer was found, scan the last 4 KB of the file.
            scan_region = trailing_bytes if trailing_bytes is not None else file_data[-4096:]
            found_urls = [m.group(0).decode('ascii', errors='replace')
                          for m in _URL_RE.finditer(scan_region)]
            if found_urls:
                result['trailing_urls'] = found_urls
                result['has_payload'] = True
                result['suspicious_indicators'].append('appended_urls')
                for url in found_urls:
                    result['warnings'].append(f'🌐 Appended URL found: {url}')
                logger.warning(f"🚨 Appended URLs detected: {found_urls}")

            # ── 3. Appended-size heuristic (original, kept intact) ────────────
            #    Compare raw file size vs. uncompressed pixel data size.

            try:
                img = Image.open(file_path)
                img_data_size = len(img.tobytes())
                overhead_threshold = 5000   # 5 KB
                if file_size > img_data_size + overhead_threshold:
                    diff = file_size - (img_data_size + overhead_threshold)
                    result['payload_size'] = max(result['payload_size'], diff)
                    result['warnings'].append(
                        f'⚠️ File has {diff:,} bytes of suspicious appended data'
                    )
                    result['has_payload'] = True
                    result['suspicious_indicators'].append('appended_data_heuristic')
                    logger.warning(f"🚨 Appended-data heuristic: {diff:,} extra bytes")
            except Exception as e:
                logger.debug(f"Could not open as image for size heuristic: {e}")

            # ── 4. Embedded file-type signatures (original, kept intact) ──────

            # ZIP
            for sig in (b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08'):
                if sig in file_data:
                    offset = file_data.find(sig)
                    result['warnings'].append(
                        f'📦 ZIP archive signature at offset {offset:,}'
                    )
                    result['has_payload'] = True
                    result['suspicious_indicators'].append('embedded_zip')
                    logger.warning(f"🚨 ZIP signature at offset {offset}")

            # PE/EXE
            if b'MZ' in file_data:
                offset = file_data.find(b'MZ')
                result['warnings'].append(
                    f'⚙️ PE/EXE signature at offset {offset:,}'
                )
                result['has_payload'] = True
                result['suspicious_indicators'].append('hidden_exe')
                logger.warning(f"🚨 EXE signature at offset {offset}")

            # PDF
            if b'%PDF' in file_data:
                offset = file_data.find(b'%PDF')
                result['warnings'].append(
                    f'📄 PDF signature at offset {offset:,}'
                )
                result['has_payload'] = True
                result['suspicious_indicators'].append('embedded_pdf')

            # ── 5. Multiple end-markers (original, kept intact) ───────────────

            iend_count = file_data.count(b'IEND')
            if iend_count > 1:
                result['warnings'].append(
                    f'🔴 Multiple IEND markers ({iend_count}) – possible appended data'
                )
                result['has_payload'] = True
                result['suspicious_indicators'].append('multiple_iend')
                logger.warning(f"🚨 Multiple IEND: {iend_count}")

            soi_count = file_data.count(b'\xff\xd8')
            if soi_count > 1:
                result['warnings'].append(
                    f'🖼️ Multiple JPEG SOI markers ({soi_count}) – possible image stitching'
                )
                result['has_payload'] = True
                result['suspicious_indicators'].append('multiple_soi')

            # ── 6. Double-extension (IMPROVED regex) ─────────────────────────
            #    Old: only matched .jpg/.jpeg/.png/.gif as the OUTER extension.
            #    New: matches any outer extension after a known image extension,
            #         e.g. photo.jpg.exe, invoice.png.php, logo.gif.js

            if _DOUBLE_EXT_RE.search(file_path):
                result['warnings'].append(
                    '⚠️ Double extension detected – file may be masquerading '
                    f'(e.g. .jpg.exe, .png.php): {os.path.basename(file_path)}'
                )
                result['has_payload'] = True
                result['suspicious_indicators'].append('double_extension')

        except Exception as e:
            logger.warning(f"Hidden payload detection failed: {e}")

        # De-duplicate indicators (multiple checks can add the same tag)
        result['suspicious_indicators'] = list(dict.fromkeys(result['suspicious_indicators']))

        if result['has_payload']:
            logger.warning(f"🚨 HIDDEN PAYLOAD DETECTED: {result['suspicious_indicators']}")

        return result

    # ── NEW: Lightweight steganalysis ─────────────────────────────────────────

    def _steganalysis(self, image_path: str) -> Dict[str, Any]:
        """
        Lightweight statistical steganalysis.  No ML model required.

        Two complementary heuristics are run:

        Chi-square attack
        -----------------
        In a natural (unmodified) image, pairs of adjacent pixel values
        (2k, 2k+1) have different frequencies.  LSB steganography equalises
        these pair frequencies because it writes random bits into the LSB,
        driving the distribution toward uniformity.  We compute the chi-square
        statistic per channel and derive a p-value; a high p-value (close to 1)
        means the LSBs are suspiciously uniform.

        RS (Regular-Singular) analysis
        --------------------------------
        Divide pixels into small groups (quads).  Apply a discrimination
        function f that measures local smoothness.  Flip the LSB and
        re-measure.  Natural images have R > S; steganographic images show
        R ≈ S (LSB flipping doesn't hurt smoothness because it was already
        random).  The RS ratio |R-S|/(R+S) approaches 0 as hidden-data
        fraction grows.

        Both methods work on numpy arrays only – no external model needed.
        """
        result = {
            'chi_square': None,
            'rs_analysis': None,
            'steg_suspected': False,
            'confidence': 'none',
            'warnings': [],
        }

        try:
            import numpy as np
        except ImportError:
            result['error'] = 'numpy not installed – steganalysis skipped'
            return result

        try:
            img = Image.open(image_path).convert('RGB')
            arr = np.array(img, dtype=np.uint8)   # shape: (H, W, 3)
        except Exception as e:
            result['error'] = str(e)
            return result

        # ── Chi-square attack ──────────────────────────────────────────────────
        chi_results = self._chi_square_attack(arr)
        result['chi_square'] = chi_results

        # ── RS analysis ───────────────────────────────────────────────────────
        rs_results = self._rs_analysis(arr)
        result['rs_analysis'] = rs_results

        # ── Combine signals ───────────────────────────────────────────────────
        chi_flag = chi_results.get('suspicious', False)
        rs_flag  = rs_results.get('suspicious', False)

        if chi_flag and rs_flag:
            result['steg_suspected'] = True
            result['confidence']     = 'high'
            result['warnings'].append(
                '🕵️ Both chi-square AND RS analysis flag potential steganography'
            )
        elif chi_flag or rs_flag:
            result['steg_suspected'] = True
            result['confidence']     = 'medium'
            method = 'chi-square' if chi_flag else 'RS analysis'
            result['warnings'].append(
                f'🕵️ {method} flags potential steganography (single-method signal)'
            )

        if result['steg_suspected']:
            logger.warning(
                f"🚨 Steg suspected (confidence={result['confidence']}): {image_path}"
            )

        return result

    def _chi_square_attack(self, arr) -> Dict[str, Any]:
        """
        Chi-square attack on LSB plane.

        For each colour channel, compute frequency of each byte value 0-255.
        Pair values (0,1), (2,3), (4,5), ... (254,255).  Under H0 (natural),
        pair members have different frequencies.  Under H1 (LSB steg), they
        converge.

        We compute the chi-square statistic and approximate the p-value using
        the regularised incomplete gamma function (available in Python's math
        module since 3.x).
        """
        import numpy as np
        import math

        channel_names = ['R', 'G', 'B']
        channel_results = {}
        suspicious_count = 0

        for c, name in enumerate(channel_names):
            channel = arr[:, :, c].flatten()
            freq = np.bincount(channel, minlength=256).astype(float)

            # Build expected frequencies: each pair expected to have equal share
            expected = np.zeros(256)
            for k in range(0, 256, 2):
                pair_total = freq[k] + freq[k + 1]
                expected[k] = expected[k + 1] = pair_total / 2.0

            # Avoid division by zero
            mask = expected > 0
            chi_sq = float(np.sum(
                ((freq[mask] - expected[mask]) ** 2) / expected[mask]
            ))

            # Degrees of freedom = number of pairs - 1 = 127
            df = 127
            # p-value via regularised upper incomplete gamma:
            #   p = 1 - gamma_lower(df/2, chi_sq/2) / Gamma(df/2)
            # math.gammainc gives the regularised lower incomplete gamma.
            try:
                p_value = 1.0 - _regularised_gamma_lower(df / 2, chi_sq / 2)
            except Exception:
                p_value = 0.0

            suspicious = p_value > 0.95   # very uniform → suspicious
            if suspicious:
                suspicious_count += 1

            channel_results[name] = {
                'chi_sq':    round(chi_sq, 2),
                'p_value':   round(p_value, 4),
                'suspicious': suspicious,
            }

        return {
            'channels':    channel_results,
            'suspicious':  suspicious_count >= 2,   # flag if ≥2 channels triggered
            'note': (
                'p_value close to 1.0 = LSBs suspiciously uniform = possible steg'
            ),
        }

    def _rs_analysis(self, arr) -> Dict[str, Any]:
        """
        RS (Regular-Singular) analysis.

        For each channel:
          1. Tile pixels into non-overlapping quads.
          2. Compute discrimination function f = sum of |p[i] - p[i+1]| per quad.
          3. Count Regular (f increases after LSB flip) and Singular (f decreases).
          4. For natural images R >> S.  Steganography drives R ≈ S.

        RS_ratio = |R - S| / (R + S)
          ~1.0  → clean image
          ~0.0  → heavy steganography

        Threshold: RS_ratio < 0.1 → suspicious.
        """
        import numpy as np

        channel_names = ['R', 'G', 'B']
        channel_results = {}
        suspicious_count = 0

        for c, name in enumerate(channel_names):
            channel = arr[:, :, c]
            H, W    = channel.shape

            # Trim to multiple of 4 columns for clean quad-tiling
            W4 = (W // 4) * 4
            if W4 == 0:
                channel_results[name] = {'error': 'image too narrow'}
                continue

            pixels = channel[:, :W4].reshape(-1, 4).astype(np.int16)

            def discriminate(p):
                """Local smoothness: sum of absolute differences between neighbours."""
                return (
                    np.abs(p[:, 0] - p[:, 1]) +
                    np.abs(p[:, 1] - p[:, 2]) +
                    np.abs(p[:, 2] - p[:, 3])
                )

            def flip_lsb(p):
                """Toggle LSB of every pixel in the group."""
                return p ^ 1

            f_orig  = discriminate(pixels)
            f_flip  = discriminate(flip_lsb(pixels))

            R = np.sum(f_flip > f_orig)   # Regular:  flipping increased roughness
            S = np.sum(f_flip < f_orig)   # Singular: flipping decreased roughness
            RS_total = R + S

            if RS_total == 0:
                channel_results[name] = {'error': 'degenerate (flat image?)'}
                continue

            rs_ratio  = abs(R - S) / RS_total
            suspicious = rs_ratio < 0.10   # threshold: near-equal R and S

            if suspicious:
                suspicious_count += 1

            channel_results[name] = {
                'R':          int(R),
                'S':          int(S),
                'rs_ratio':   round(float(rs_ratio), 4),
                'suspicious': suspicious,
                'note': 'rs_ratio near 0 = R≈S = possible steg; near 1 = clean',
            }

        return {
            'channels':  channel_results,
            'suspicious': suspicious_count >= 2,
        }


# ── Module-level helper: regularised lower incomplete gamma ───────────────────

def _regularised_gamma_lower(a: float, x: float) -> float:
    """
    Regularised lower incomplete gamma P(a, x) using the series expansion.
    Only needed for the chi-square p-value.  Pure Python – no scipy required.
    """
    import math
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0

    # Series: P(a,x) = e^(-x) * x^a * sum_{n=0}^{inf} x^n / Gamma(a+n+1)
    MAX_ITER = 200
    TOL      = 1e-10
    term     = 1.0 / a
    total    = term
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