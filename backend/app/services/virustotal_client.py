# app/services/virustotal_client.py
import os
import time
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

VIRUSTOTAL_API_URL = "https://www.virustotal.com/api/v3"
VIRUSTOTAL_API_KEY = os.environ.get("VIRUSTOTAL_API_KEY", "")


class VirusTotalClient:
    """
    Lightweight VirusTotal API v3 client.
    Supports URL and file analysis.
    """

    def __init__(self):
        self.api_key = VIRUSTOTAL_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "x-apikey": self.api_key,
            "Accept": "application/json",
        })
        if self.api_key:
            logger.info("✅ VirusTotal client initialised")
        else:
            logger.warning("⚠️  No VirusTotal API key found — VT module will not work")

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict]:
        """Base request with error handling."""
        if not self.api_key:
            return {"error": "VirusTotal API key not configured"}
        try:
            url = f"{VIRUSTOTAL_API_URL}{endpoint}"
            resp = self.session.request(method, url, timeout=15, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            logger.error(f"VirusTotal HTTP error: {e}")
            try:
                detail = resp.json().get("error", {}).get("message", str(e))
            except Exception:
                detail = str(e)
            return {"error": detail}
        except Exception as e:
            logger.error(f"VirusTotal request failed: {e}")
            return {"error": str(e)}

    def check_url(self, url: str) -> Dict[str, Any]:
        """
        Analyse a URL using VirusTotal.
        Returns a flat dict suitable for the orchestrator.
        """
        # Submit URL for analysis
        submit_result = self._request("POST", "/urls", data={"url": url})
        if not submit_result or "error" in submit_result:
            return self._error_result(submit_result.get("error", "Submission failed"))

        analysis_id = None
        try:
            analysis_id = submit_result["data"]["id"]
        except KeyError:
            return self._error_result("Invalid response from VirusTotal")

        # Wait briefly for analysis to complete (VT may already have cached result)
        time.sleep(2)

        # Retrieve analysis result
        result = self._request("GET", f"/analyses/{analysis_id}")
        if not result or "error" in result:
            return self._error_result(result.get("error", "Analysis retrieval failed"))

        return self._parse_url_result(result)

    def check_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload a file and retrieve its VirusTotal report.
        """
        if not os.path.exists(file_path):
            return self._error_result("File not found")

        # Get upload URL
        upload_url_info = self._request("GET", "/files/upload_url")
        if not upload_url_info or "error" in upload_url_info:
            return self._error_result(upload_url_info.get("error", "Upload URL failed"))

        upload_url = None
        try:
            upload_url = upload_url_info["data"]
        except KeyError:
            return self._error_result("Invalid upload URL response")

        # Upload file
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            upload_resp = self.session.post(upload_url, files=files, timeout=30)
            try:
                upload_resp.raise_for_status()
                file_data = upload_resp.json()
            except Exception as e:
                return self._error_result(f"File upload failed: {e}")

        analysis_id = None
        try:
            analysis_id = file_data["data"]["id"]
        except KeyError:
            return self._error_result("Invalid file analysis ID")

        # Wait for analysis
        time.sleep(5)

        # Retrieve analysis
        result = self._request("GET", f"/analyses/{analysis_id}")
        if not result or "error" in result:
            return self._error_result(result.get("error", "Analysis retrieval failed"))

        return self._parse_file_result(result)

    def _parse_url_result(self, data: Dict) -> Dict[str, Any]:
        """Parse URL analysis response into flat verdict."""
        attributes = data.get("data", {}).get("attributes", {})
        stats = attributes.get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        undetected = stats.get("undetected", 0)
        total = malicious + suspicious + undetected
        verdict = "Clean"
        if malicious > 0:
            verdict = "Malicious"
        elif suspicious > 0:
            verdict = "Suspicious"

        return {
            "success": True,
            "verdict": verdict,
            "malicious": malicious,
            "suspicious": suspicious,
            "undetected": undetected,
            "total_engines": total,
            "score": malicious * 10 + suspicious * 5,
            "permalink": f"https://www.virustotal.com/gui/url/{data['data']['id'].split('-')[1]}",
            "details": attributes.get("last_analysis_results", {}),
        }

    def _parse_file_result(self, data: Dict) -> Dict[str, Any]:
        """Parse file analysis response."""
        attributes = data.get("data", {}).get("attributes", {})
        stats = attributes.get("last_analysis_stats", {})
        malicious = stats.get("malicious", 0)
        suspicious = stats.get("suspicious", 0)
        undetected = stats.get("undetected", 0)
        total = malicious + suspicious + undetected
        verdict = "Clean"
        if malicious > 0:
            verdict = "Malicious"
        elif suspicious > 0:
            verdict = "Suspicious"

        return {
            "success": True,
            "verdict": verdict,
            "malicious": malicious,
            "suspicious": suspicious,
            "undetected": undetected,
            "total_engines": total,
            "score": malicious * 10 + suspicious * 5,
            "permalink": f"https://www.virustotal.com/gui/file/{data['data']['id'].split('-')[1]}",
            "details": attributes.get("last_analysis_results", {}),
        }

    def _error_result(self, error: str) -> Dict[str, Any]:
        return {
            "success": False,
            "error": error,
            "verdict": "Unknown",
            "malicious": 0,
            "suspicious": 0,
            "undetected": 0,
            "score": 0,
            "permalink": None,
        }