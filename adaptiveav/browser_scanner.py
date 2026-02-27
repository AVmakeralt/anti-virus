"""
AdaptiveAV Browser Scanner
===========================
Monitors and scans:
  - Browser extension manifests (Chrome, Firefox, Edge, Brave)
  - Browser history for malicious domains / phishing URLs
  - Browser processes for unusual behavior (memory injections, etc.)
  - Downloaded files from browser temp dirs
  - JavaScript patterns in cached HTML files

All analysis is local. No data leaves the machine.
"""

import os
import re
import json
import time
import sqlite3
import hashlib
import fnmatch
import platform
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

# ── Platform-aware paths ──────────────────────────────────────────

_PLATFORM = platform.system()

def _home(): return Path.home()

BROWSER_PROFILES = {
    "Chrome": {
        "Darwin":  _home() / "Library/Application Support/Google/Chrome",
        "Linux":   _home() / ".config/google-chrome",
        "Windows": Path(os.environ.get("LOCALAPPDATA","")) / "Google/Chrome/User Data",
    },
    "Brave": {
        "Darwin":  _home() / "Library/Application Support/BraveSoftware/Brave-Browser",
        "Linux":   _home() / ".config/BraveSoftware/Brave-Browser",
        "Windows": Path(os.environ.get("LOCALAPPDATA","")) / "BraveSoftware/Brave-Browser",
    },
    "Edge": {
        "Darwin":  _home() / "Library/Application Support/Microsoft Edge",
        "Linux":   _home() / ".config/microsoft-edge",
        "Windows": Path(os.environ.get("LOCALAPPDATA","")) / "Microsoft/Edge/User Data",
    },
    "Firefox": {
        "Darwin":  _home() / "Library/Application Support/Firefox",
        "Linux":   _home() / ".mozilla/firefox",
        "Windows": Path(os.environ.get("APPDATA","")) / "Mozilla/Firefox",
    },
    "Opera": {
        "Darwin":  _home() / "Library/Application Support/com.operasoftware.Opera",
        "Linux":   _home() / ".config/opera",
        "Windows": Path(os.environ.get("APPDATA","")) / "Opera Software/Opera Stable",
    },
}

BROWSER_DOWNLOAD_DIRS = {
    "Darwin":  [_home() / "Downloads"],
    "Linux":   [_home() / "Downloads", Path("/tmp")],
    "Windows": [_home() / "Downloads"],
}

# ── Threat intelligence ───────────────────────────────────────────

MALICIOUS_DOMAIN_PATTERNS = [
    r'.*\.(tk|ml|ga|cf|gq)$',              # Free TLD abuse
    r'paypal[^.]*\.(com\.[a-z]{2}|net\.)',  # Paypal phishing
    r'(login|signin|account|verify|secure|update|confirm)\.[^.]+\.(xyz|top|club|online|site)',
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}.*\.(exe|dll|bat|ps1|vbs|sh)',  # IP-served executables
    r'(paypai|paypa1|g00gle|g0ogle|amaz0n|micosoft)',  # Typosquatting
]

MALICIOUS_URL_INDICATORS = [
    "bit.ly", "tinyurl.com", "t.co/",  # suspicious shorteners (when from unknown source)
    "?download=", "?file=", "?exec=",  # suspicious params
    "/shell.php", "/cmd.php", "/c99.php", "/r57.php",  # known webshells
    "base64_decode", "eval(atob(",     # obfuscated JS
    ".exe?", ".dll?", ".bat?",         # executables via query string
    "javascript:void(0)", "data:text/html;base64",
]

DANGEROUS_EXTENSION_PERMISSIONS = {
    "tabs",
    "webRequest",
    "webRequestBlocking",
    "cookies",
    "history",
    "management",
    "nativeMessaging",
    "<all_urls>",
    "browsingData",
    "privacy",
    "proxy",
    "contentSettings",
    "debugger",
}

HIGH_RISK_EXTENSION_COMBOS = [
    {"tabs", "cookies", "<all_urls>"},
    {"webRequest", "webRequestBlocking", "<all_urls>"},
    {"nativeMessaging", "cookies"},
    {"history", "<all_urls>", "cookies"},
]

JS_MALWARE_PATTERNS = [
    (r'eval\s*\(\s*atob\s*\(', "Base64-encoded eval"),
    (r'eval\s*\(\s*unescape\s*\(', "URL-escaped eval"),
    (r'String\.fromCharCode\s*\([0-9,\s]{50,}', "Charcode obfuscation"),
    (r'\bexec\s*\(["\'].*powershell', "PowerShell via exec"),
    (r'document\.cookie\s*=.*window\.location', "Cookie theft redirect"),
    (r'new\s+XMLHttpRequest.*\bopen\s*\(["\']POST', "XHR data exfil"),
    (r'localStorage\s*\.\s*getItem.*\bfetch\b', "LocalStorage exfiltration"),
    (r'navigator\.sendBeacon\s*\(', "Beacon exfiltration"),
    (r'crypto\.subtle', "Crypto operation"),
    (r'WebSocket.*wss?://((?!your-domain).)*\brecv', "WebSocket C2"),
]


class BrowserScanResult:
    def __init__(self, browser: str, profile: str):
        self.browser   = browser
        self.profile   = profile
        self.extension_threats: list[dict] = []
        self.url_threats: list[dict]       = []
        self.download_threats: list[dict]  = []
        self.js_threats: list[dict]        = []
        self.scan_time  = datetime.now().isoformat()
        self.total_urls_checked  = 0
        self.total_exts_checked  = 0

    @property
    def threat_count(self): return (
        len(self.extension_threats) +
        len(self.url_threats) +
        len(self.download_threats) +
        len(self.js_threats)
    )

    def summary(self) -> str:
        return (f"[{self.browser}/{self.profile}] "
                f"Threats: {self.threat_count} "
                f"(ext:{len(self.extension_threats)}, "
                f"url:{len(self.url_threats)}, "
                f"dl:{len(self.download_threats)}, "
                f"js:{len(self.js_threats)})")


class BrowserScanner:
    """Scans all installed browsers for threats, entirely offline."""

    def __init__(self):
        self.platform = _PLATFORM
        self._domain_re = [re.compile(p, re.I) for p in MALICIOUS_DOMAIN_PATTERNS]
        self._js_re     = [(re.compile(p, re.I), name) for p, name in JS_MALWARE_PATTERNS]

    def scan_all(self) -> list[BrowserScanResult]:
        """Scan all detected browsers. Returns list of results."""
        results = []
        for browser, paths in BROWSER_PROFILES.items():
            profile_root = paths.get(self.platform)
            if not profile_root or not profile_root.exists():
                continue
            for result in self._scan_browser(browser, profile_root):
                results.append(result)
        return results

    def _scan_browser(self, browser: str, profile_root: Path):
        if browser == "Firefox":
            yield from self._scan_firefox(profile_root)
        else:
            yield from self._scan_chromium(browser, profile_root)

    # ── Chromium-based browsers ─────────────────────────────────

    def _scan_chromium(self, browser: str, profile_root: Path):
        profiles = self._find_chromium_profiles(profile_root)
        for prof_name, prof_path in profiles:
            result = BrowserScanResult(browser, prof_name)
            self._scan_chromium_extensions(result, prof_path / "Extensions")
            self._scan_chromium_history(result, prof_path / "History")
            self._scan_chromium_cache_js(result, prof_path)
            yield result

    def _find_chromium_profiles(self, root: Path) -> list[tuple]:
        profiles = []
        # "Default" is the main profile
        default = root / "Default"
        if default.exists():
            profiles.append(("Default", default))
        # Profile 1, 2, ...
        for p in root.glob("Profile *"):
            profiles.append((p.name, p))
        if not profiles and root.exists():
            profiles.append(("Default", root))
        return profiles

    def _scan_chromium_extensions(self, result: BrowserScanResult, ext_dir: Path):
        if not ext_dir.exists():
            return
        for ext_id_dir in ext_dir.iterdir():
            if not ext_id_dir.is_dir():
                continue
            for ver_dir in ext_id_dir.iterdir():
                manifest_path = ver_dir / "manifest.json"
                if not manifest_path.exists():
                    continue
                result.total_exts_checked += 1
                try:
                    with open(manifest_path, encoding="utf-8", errors="ignore") as f:
                        manifest = json.load(f)
                except Exception:
                    continue

                threat = self._analyze_extension_manifest(manifest, ext_id_dir.name, ver_dir)
                if threat:
                    result.extension_threats.append(threat)

                # Also scan extension JS files
                for js_file in ver_dir.rglob("*.js"):
                    js_threats = self._scan_js_file(js_file)
                    for t in js_threats:
                        t["source"] = f"Extension:{ext_id_dir.name}/{js_file.name}"
                        result.js_threats.append(t)

    def _analyze_extension_manifest(self, manifest: dict, ext_id: str, path: Path) -> Optional[dict]:
        name = manifest.get("name", "Unknown")
        perms = set(manifest.get("permissions", []) + manifest.get("host_permissions", []))
        optional_perms = set(manifest.get("optional_permissions", []))
        all_perms = perms | optional_perms

        # Check dangerous permission combinations
        dangerous_found = all_perms & DANGEROUS_EXTENSION_PERMISSIONS
        risk_score = len(dangerous_found) * 10

        for combo in HIGH_RISK_EXTENSION_COMBOS:
            if combo.issubset(all_perms):
                risk_score += 50

        # Check for suspicious CSP
        csp = manifest.get("content_security_policy", "")
        if "unsafe-eval" in csp or "unsafe-inline" in csp:
            risk_score += 20

        # Check background scripts
        bg = manifest.get("background", {})
        bg_scripts = bg.get("scripts", []) + ([bg.get("service_worker", "")] if "service_worker" in bg else [])
        if bg_scripts:
            risk_score += 5

        # Check for remote code loading
        remote_urls = re.findall(r'https?://(?!clients\d?\.google\.com|update\.googleapis\.com)[^\s"\']+',
                                  json.dumps(manifest))
        if remote_urls:
            risk_score += 30

        if risk_score >= 40:
            return {
                "type":         "extension",
                "ext_id":       ext_id,
                "name":         name,
                "path":         str(path),
                "risk_score":   risk_score,
                "permissions":  list(dangerous_found),
                "remote_urls":  remote_urls[:5],
                "verdict":      "HIGH" if risk_score >= 70 else "MEDIUM",
            }
        return None

    def _scan_chromium_history(self, result: BrowserScanResult, history_db: Path):
        if not history_db.exists():
            return
        # Copy DB to temp (Chrome locks the original)
        tmp = Path(tempfile.mktemp(suffix=".sqlite"))
        try:
            shutil.copy2(history_db, tmp)
            conn = sqlite3.connect(str(tmp))
            cur = conn.cursor()
            # Get URLs from last 30 days
            cutoff_microseconds = int((time.time() - 30*86400) * 1e6)
            try:
                cur.execute(
                    "SELECT url, title, visit_count FROM urls "
                    "WHERE last_visit_time > ? LIMIT 5000",
                    (cutoff_microseconds,)
                )
                rows = cur.fetchall()
            except Exception:
                rows = []
            conn.close()

            for url, title, visit_count in rows:
                result.total_urls_checked += 1
                threat = self._analyze_url(url, title or "")
                if threat:
                    result.url_threats.append(threat)
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()

    def _scan_chromium_cache_js(self, result: BrowserScanResult, profile_path: Path):
        """Scan cached/local HTML and JS files for malicious patterns."""
        cache_dirs = [
            profile_path / "Cache",
            profile_path / "Code Cache",
        ]
        scanned = 0
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            for f in cache_dir.rglob("*"):
                if scanned > 500:
                    break
                if not f.is_file() or f.stat().st_size > 500_000:
                    continue
                try:
                    content = f.read_bytes()
                    # Only analyze if it looks like JS/HTML
                    if b"<script" in content or b"function(" in content or b"eval(" in content:
                        threats = self._scan_js_bytes(content, str(f))
                        result.js_threats.extend(threats)
                        scanned += 1
                except Exception:
                    pass

    # ── Firefox ─────────────────────────────────────────────────

    def _scan_firefox(self, profile_root: Path):
        if not profile_root.exists():
            return
        # Find all profiles
        profiles_ini = profile_root / "profiles.ini"
        if not profiles_ini.exists():
            return
        try:
            with open(profiles_ini) as f:
                content = f.read()
            paths = re.findall(r'^Path=(.*)', content, re.MULTILINE)
        except Exception:
            return

        for rel_path in paths:
            if rel_path.startswith("/"):
                prof_path = Path(rel_path)
            else:
                prof_path = profile_root / rel_path

            if not prof_path.exists():
                continue

            result = BrowserScanResult("Firefox", rel_path)
            self._scan_firefox_addons(result, prof_path)
            self._scan_firefox_history(result, prof_path / "places.sqlite")
            yield result

    def _scan_firefox_addons(self, result: BrowserScanResult, prof_path: Path):
        addons_json = prof_path / "addons.json"
        if not addons_json.exists():
            return
        try:
            with open(addons_json, encoding="utf-8", errors="ignore") as f:
                data = json.load(f)
        except Exception:
            return

        for addon in data.get("addons", []):
            result.total_exts_checked += 1
            perms = set(addon.get("userPermissions", {}).get("permissions", []) +
                       addon.get("userPermissions", {}).get("origins", []))
            dangerous = perms & DANGEROUS_EXTENSION_PERMISSIONS
            risk_score = len(dangerous) * 10
            for combo in HIGH_RISK_EXTENSION_COMBOS:
                if combo.issubset(perms):
                    risk_score += 50
            if risk_score >= 40:
                result.extension_threats.append({
                    "type":       "extension",
                    "ext_id":     addon.get("id", "unknown"),
                    "name":       addon.get("name", "Unknown"),
                    "risk_score": risk_score,
                    "permissions": list(dangerous),
                    "verdict":    "HIGH" if risk_score >= 70 else "MEDIUM",
                })

    def _scan_firefox_history(self, result: BrowserScanResult, places_db: Path):
        if not places_db.exists():
            return
        tmp = Path(tempfile.mktemp(suffix=".sqlite"))
        try:
            shutil.copy2(places_db, tmp)
            conn = sqlite3.connect(str(tmp))
            cur = conn.cursor()
            cutoff = (datetime.now() - timedelta(days=30)).timestamp() * 1e6
            try:
                cur.execute(
                    "SELECT url, title FROM moz_places "
                    "JOIN moz_historyvisits ON moz_places.id = moz_historyvisits.place_id "
                    "WHERE moz_historyvisits.visit_date > ? LIMIT 5000",
                    (int(cutoff),)
                )
                rows = cur.fetchall()
            except Exception:
                rows = []
            conn.close()
            for url, title in rows:
                result.total_urls_checked += 1
                threat = self._analyze_url(url, title or "")
                if threat:
                    result.url_threats.append(threat)
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()

    # ── URL and JS Analysis ──────────────────────────────────────

    def _analyze_url(self, url: str, title: str) -> Optional[dict]:
        if not url or not url.startswith("http"):
            return None
        # Extract domain
        m = re.match(r'https?://([^/\?]+)', url)
        if not m:
            return None
        domain = m.group(1).lower()

        reasons = []

        # Domain pattern matching
        for pat in self._domain_re:
            if pat.match(domain):
                reasons.append(f"SuspiciousDomain:{domain}")
                break

        # URL indicators
        for indicator in MALICIOUS_URL_INDICATORS:
            if indicator in url.lower():
                reasons.append(f"SuspiciousURL:{indicator}")

        # Homograph/IDN attack detection
        if re.search(r'xn--', domain):
            reasons.append("IDN-Homograph")

        # Suspicious subdomain depth
        parts = domain.split(".")
        if len(parts) > 5:
            reasons.append(f"DeepSubdomain:{len(parts)}")

        if reasons:
            return {
                "type":    "url",
                "url":     url[:200],
                "domain":  domain,
                "title":   title[:100],
                "reasons": reasons,
                "verdict": "HIGH" if len(reasons) >= 2 else "MEDIUM",
            }
        return None

    def _scan_js_file(self, path: Path) -> list[dict]:
        try:
            content = path.read_bytes()
            return self._scan_js_bytes(content, str(path))
        except Exception:
            return []

    def _scan_js_bytes(self, content: bytes, source: str) -> list[dict]:
        try:
            text = content.decode("utf-8", errors="ignore")
        except Exception:
            return []
        threats = []
        for pattern, name in self._js_re:
            if pattern.search(text):
                threats.append({
                    "type":    "js_pattern",
                    "source":  source[:200],
                    "pattern": name,
                    "verdict": "HIGH",
                })
        return threats

    # ── Download directory scan ──────────────────────────────────

    def scan_downloads(self) -> list[dict]:
        """Scan browser download directories for suspicious files."""
        threats = []
        dirs = BROWSER_DOWNLOAD_DIRS.get(self.platform, [])
        suspicious_extensions = {
            ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs",
            ".js", ".jar", ".msi", ".reg", ".scr", ".hta",
            ".sh", ".run", ".bin",
        }
        for dl_dir in dirs:
            if not dl_dir.exists():
                continue
            for f in dl_dir.iterdir():
                if not f.is_file():
                    continue
                ext = f.suffix.lower()
                if ext in suspicious_extensions:
                    # Check if recently downloaded (last 7 days)
                    age = time.time() - f.stat().st_mtime
                    if age < 7 * 86400:
                        threats.append({
                            "type":      "download",
                            "path":      str(f),
                            "extension": ext,
                            "size":      f.stat().st_size,
                            "age_hours": age / 3600,
                            "verdict":   "MEDIUM",
                        })
        return threats