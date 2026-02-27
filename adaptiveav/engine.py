"""
AdaptiveAV Engine v3 — Truly Adaptive
=======================================
Changes from v2:
  - Replaced fake AdaptiveClassifier with TrulyAdaptiveClassifier:
      • Full backprop (Adam, all layers, gradient clipping)
      • Prioritized experience replay buffer (2000 samples)
      • Page-Hinkley concept drift detection → auto-replays on drift
      • Monte Carlo Dropout uncertainty quantification
      • Learned feature importance scaling (per-feature gradient magnitudes)
      • Dynamic ensemble meta-learner (per file-type-cluster weights)
      • Adaptive heuristic rule weights via credit assignment
      • Welford online mean/variance NB with exponential forgetting
  - scan() now passes features dict and fired_rules to learn()
  - Auto-learning uses calibrated confidence + uncertainty gating
  - report() shows per-component accuracy, top features, drift count
  - teach() uses proper weighted learning (user labels = high weight)
"""

import os
import sys
import time
import json
import math
import hashlib
import random
import statistics
import platform
import re
import pickle
from pathlib import Path
from collections import Counter
from typing import Optional

# package imports (no path hacks required once inside adaptiveav)
from adaptiveav.isolation import QuarantineManager, UserInstalledAppRegistry, SandboxManager
from adaptiveav.browser_scanner import BrowserScanner, BrowserScanResult
from adaptiveav.file_watch import AVDaemon, AlertBus
from adaptiveav.adaptive import TrulyAdaptiveClassifier, FEATURE_NAMES

# ── ANSI Output ───────────────────────────────────────────────────

class C:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    CYAN    = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE   = "\033[97m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def cprint(color, text): print(f"{color}{text}{C.RESET}")

ADAPTIVEAV_DIR = Path.home() / ".adaptiveav"
MODEL_CACHE    = ADAPTIVEAV_DIR / "engine_v3_state.pkl"


# ── Feature Extractor ─────────────────────────────────────────────

class FeatureExtractor:
    MALWARE_SIGNATURES = {
        b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR": "EICAR-Test",
        b"powershell -enc":         "PS-Encoded-Command",
        b"cmd.exe /c":              "CMD-Execution",
        b"WScript.Shell":           "VBS-Shell",
        b"CreateRemoteThread":      "Process-Injection",
        b"VirtualAllocEx":          "Memory-Injection",
        b"WriteProcessMemory":      "Process-Memory-Write",
        b"InternetConnect":         "Network-Backdoor",
        b"base64.b64decode":        "Base64-Payload",
        b"eval(compile(":           "Python-Code-Eval",
        b"__import__('os').system": "Python-OS-Exec",
        b"subprocess.Popen":        "Python-Subprocess",
        b"os.system(":              "OS-System-Call",
        b"/bin/sh -i":              "Reverse-Shell",
        b"nc -e /bin/":             "Netcat-Shell",
        b"chmod 777":               "Permissions-Escalation",
        b"nohup ":                  "Background-Execution",
        b"crontab -":               "Cron-Persistence",
        b"LD_PRELOAD":              "Library-Injection",
        b"ptrace":                  "Debug-Hook",
        b"msfvenom":                "Metasploit-Payload",
        b"meterpreter":             "Meterpreter-Shell",
        b"mimikatz":                "Mimikatz-Credential-Dump",
    }

    # initial suspicious terms, but list can grow dynamically
    SUSPICIOUS_STRINGS = [
        "password", "keylogger", "ransomware", "exploit", "shellcode",
        "payload", "backdoor", "rootkit", "trojan", "dropper", "downloader",
        "encrypt_files", "delete_shadow", "disable_antivirus",
        "bypass_uac", "privilege_escalation", "steal", "exfiltrate",
        "c2server", "command_and_control", "reverse_shell", "bind_shell",
    ]

    SCRIPT_EXTS = {'.py', '.js', '.vbs', '.ps1', '.bat', '.sh', '.php', '.rb', '.pl', '.lua'}

    def __init__(self):
        # running statistics for numeric features (Welford)
        self._numeric_stats = {}  # name -> (mean, m2, count)
        # thresholds computed on the fly
        self._thresholds = {
            'entropy_high': 7.0,
            'entropy_very_high': 7.5,
        }

    # ----- util for online stats -----
    def _update_numeric(self, name: str, value: float):
        mean, m2, n = self._numeric_stats.get(name, (0.0, 0.0, 0))
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
        self._numeric_stats[name] = (mean, m2, n)
        std = math.sqrt(m2 / n) if n > 1 else 0.0
        # adapt thresholds for entropy
        if name == 'entropy':
            self._thresholds['entropy_high'] = mean + std * 1.0
            self._thresholds['entropy_very_high'] = mean + std * 2.0
        return mean, std

    # ----- persistence -----
    def state_dict(self) -> dict:
        return {
            '_numeric_stats': self._numeric_stats,
            '_thresholds': self._thresholds,
            'suspicious_strings': list(self.SUSPICIOUS_STRINGS),
            'script_exts': list(self.SCRIPT_EXTS),
        }

    def load_state(self, d: dict):
        if '_numeric_stats' in d:
            self._numeric_stats = d['_numeric_stats']
        if '_thresholds' in d:
            self._thresholds = d['_thresholds']
        if 'suspicious_strings' in d:
            self.SUSPICIOUS_STRINGS = list(d['suspicious_strings'])
        if 'script_exts' in d:
            self.SCRIPT_EXTS = set(d.get('script_exts', []))

    # ----- adaptation upon feedback -----
    def update_on_label(self, features: dict, path: Optional[str], label: str):
        """Called by engine when a sample is learned or user‑labeled.
        Enables the extractor to expand its heuristics over time.
        """
        # extend suspicious strings set if new ones appear in malicious files
        if label == 'malicious' and features:
            for s in features.get('suspicious_strings', []):
                if s and s not in self.SUSPICIOUS_STRINGS:
                    self.SUSPICIOUS_STRINGS.append(s)
        # add new script extensions seen in malicious samples
        if label == 'malicious' and path:
            ext = Path(path).suffix.lower()
            if ext and ext not in self.SCRIPT_EXTS:
                self.SCRIPT_EXTS.add(ext)

    def extract(self, path: str) -> dict:
        f = {"readable": False, "file_size": 0}
        try:
            f["file_size"] = os.stat(path).st_size
        except Exception:
            return f
        try:
            with open(path, "rb") as fh:
                data = fh.read(1024 * 1024)
        except Exception:
            return f

        f["readable"]  = True
        f["extension"] = Path(path).suffix.lower()
        f["is_script"] = f["extension"] in self.SCRIPT_EXTS

        bc = Counter(data)
        n  = len(data)
        f["entropy"] = -sum((c/n)*math.log2(c/n) for c in bc.values() if c > 0) if n else 0.0
        # update statistics for entropy (used to adapt thresholds)
        self._update_numeric('entropy', f['entropy'])
        f["entropy_high"] = f["entropy"] > self._thresholds.get('entropy_high', 7.0)
        f["entropy_very_high"] = f["entropy"] > self._thresholds.get('entropy_very_high', 7.5)

        f["unique_bytes"]      = len(bc)
        # track additional numerics as needed
        _ = self._update_numeric('unique_bytes', f['unique_bytes'])

        f["null_byte_ratio"]   = bc.get(0, 0) / max(n, 1)
        _ = self._update_numeric('null_byte_ratio', f['null_byte_ratio'])

        f["printable_ratio"]   = sum(bc.get(c, 0) for c in range(32, 127)) / max(n, 1)
        _ = self._update_numeric('printable_ratio', f['printable_ratio'])

        f["high_byte_ratio"]   = sum(bc.get(c, 0) for c in range(128, 256)) / max(n, 1)
        _ = self._update_numeric('high_byte_ratio', f['high_byte_ratio'])

        f["is_pe"]     = data[:2] == b"MZ"
        f["is_elf"]    = data[:4] == b"\x7fELF"
        f["is_zip"]    = data[:2] == b"PK"
        f["is_pdf"]    = data[:4] == b"%PDF"
        f["is_binary"] = f["is_pe"] or f["is_elf"]

        dl   = data.lower()
        hits = {name: True for sig, name in self.MALWARE_SIGNATURES.items() if sig.lower() in dl}
        f["signature_hits"]      = hits
        f["signature_hit_count"] = len(hits)
        f["known_signature"]     = bool(hits)

        text = data.decode("utf-8", errors="ignore").lower()
        sus  = [s for s in self.SUSPICIOUS_STRINGS if s in text]
        f["suspicious_strings"]      = sus
        f["suspicious_string_count"] = len(sus)
        f["hex_string_count"]        = text.count("0x")
        f["long_string_count"]       = sum(1 for w in text.split() if len(w) > 50)

        urls = re.findall(r'https?://[^\s"\'<>]+', text)
        ips  = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)
        f["url_count"]   = len(urls)
        f["ip_count"]    = len(ips)
        f["has_network"] = bool(urls or ips)

        if f["is_pe"] and len(data) > 64:
            chunks = [self._ent(data[i:i+256]) for i in range(0, min(len(data), 8192), 256)]
            f["section_entropy_variance"] = statistics.variance(chunks) if len(chunks) > 1 else 0.0
            f["max_section_entropy"]      = max(chunks)
        else:
            f["section_entropy_variance"] = 0.0
            f["max_section_entropy"]      = 0.0

        f["md5"]    = hashlib.md5(data).hexdigest()
        f["sha256"] = hashlib.sha256(data).hexdigest()
        return f

    def _ent(self, d: bytes) -> float:
        if not d: return 0.0
        c = Counter(d); n = len(d)
        return -sum((v/n)*math.log2(v/n) for v in c.values() if v > 0)

    def to_vector(self, f: dict) -> list:
        if not f.get("readable"):
            return [0.0] * 20
        return [
            f.get("entropy", 0) / 8.0,
            float(f.get("entropy_high", False)),
            float(f.get("entropy_very_high", False)),
            min(f.get("unique_bytes", 0) / 256.0, 1.0),
            f.get("null_byte_ratio", 0),
            f.get("printable_ratio", 0),
            f.get("high_byte_ratio", 0),
            float(f.get("is_pe", False)),
            float(f.get("is_script", False)),
            min(f.get("signature_hit_count", 0) / 5.0, 1.0),
            float(f.get("known_signature", False)),
            min(f.get("suspicious_string_count", 0) / 10.0, 1.0),
            min(f.get("long_string_count", 0) / 20.0, 1.0),
            min(f.get("hex_string_count", 0) / 50.0, 1.0),
            float(f.get("has_network", False)),
            min(f.get("url_count", 0) / 10.0, 1.0),
            min(f.get("ip_count", 0) / 10.0, 1.0),
            min(f.get("section_entropy_variance", 0) / 10.0, 1.0),
            f.get("max_section_entropy", 0) / 8.0,
            min(f.get("file_size", 0), 1e6) / 1e6,
        ]


# ── Threat Intelligence ───────────────────────────────────────────

class ThreatIntel:
    def __init__(self):
        self.malicious: dict = {}
        self.benign:    set  = set()
        self.families:  Counter = Counter()

    def add_malicious(self, sha256, name):
        self.malicious[sha256] = name
        self.families[name.split("-")[0]] += 1

    def add_benign(self, sha256):
        self.benign.add(sha256)

    def lookup(self, sha256) -> Optional[str]:
        if sha256 in self.malicious: return self.malicious[sha256]
        if sha256 in self.benign:    return "CLEAN"
        return None


# ── Scan Result ───────────────────────────────────────────────────

class ScanResult:
    def __init__(self):
        self.path             = ""
        self.verdict          = "unknown"
        self.confidence       = 0.0
        self.uncertainty      = 0.0
        self.risk_level       = "UNKNOWN"
        self.threat_name      = None
        self.detection_method = []
        self.ml_details       = {}
        self.features         = {}
        self.scan_time_ms     = 0
        self.quarantine_action = None

    def risk_color(self):
        return {
            "CRITICAL": C.RED + C.BOLD,
            "HIGH":     C.RED,
            "MEDIUM":   C.YELLOW,
            "LOW":      C.GREEN + C.DIM,
            "CLEAN":    C.GREEN + C.BOLD,
        }.get(self.risk_level, C.WHITE)


# ── Main Engine ───────────────────────────────────────────────────

class AdaptiveAVEngine:
    VERSION = "3.0.0-TrulyAdaptive"

    def __init__(self, use_large_model: bool = False):
        ADAPTIVEAV_DIR.mkdir(exist_ok=True, parents=True)

        self.extractor        = FeatureExtractor()
        self.classifier       = TrulyAdaptiveClassifier(n_features=20)
        self.intel            = ThreatIntel()
        self.quarantine       = QuarantineManager()
        self.browser_scanner  = BrowserScanner()
        self.scan_count       = 0
        self.threat_count     = 0
        self._large_model     = None
        self._use_large_model = use_large_model

        self._load_state()
        if self.classifier.n_learned < 10:
            self._bootstrap()

        if len(self.quarantine.app_registry.list_protected()) < 5:
            n = self.quarantine.app_registry.auto_detect_user_apps()
            if n > 0:
                cprint(C.DIM, f"  [Registered {n} user-installed app paths as protected]")

    # ── Core scan ────────────────────────────────────────────────

    def scan(self, path: str) -> ScanResult:
        t0  = time.time()
        res = ScanResult()
        res.path = path

        if not os.path.exists(path):
            res.verdict = "error"; res.risk_level = "UNKNOWN"
            return res

        features = self.extractor.extract(path)
        res.features = features

        if not features.get("readable"):
            res.verdict = "error"; res.risk_level = "UNKNOWN"
            res.scan_time_ms = int((time.time() - t0) * 1000)
            return res

        sha256 = features["sha256"]
        vector = self.extractor.to_vector(features)

        # Stage 1: Hash-based threat intel
        ti = self.intel.lookup(sha256)
        if ti == "CLEAN":
            res.verdict = "benign"; res.confidence = 0.99
            res.risk_level = "CLEAN"
            res.detection_method.append("TI:KnownClean")
            self.scan_count += 1
            res.scan_time_ms = int((time.time() - t0) * 1000)
            return res
        if ti:
            res.verdict = "malicious"; res.threat_name = ti
            res.confidence = 1.0; res.risk_level = "CRITICAL"
            res.detection_method.append(f"TI:{ti}")
            self.classifier.learn(vector, "malicious", features=features, weight=3.0)
            # adapt extractor heuristics
            self.extractor.update_on_label(features, path, "malicious")
            self._post_threat(res, path, sha256)
            self.scan_count += 1
            res.scan_time_ms = int((time.time() - t0) * 1000)
            return res

        # Stage 2: Signature matching
        if features.get("known_signature"):
            res.verdict     = "malicious"
            res.threat_name = "/".join(features["signature_hits"].keys())[:80]
            res.confidence  = 0.99; res.risk_level = "CRITICAL"
            res.detection_method.append("Signature")
            self.intel.add_malicious(sha256, res.threat_name)
            self.classifier.learn(
                vector, "malicious", features=features,
                fired_rules=list(features["signature_hits"].keys()), weight=5.0
            )
            self.extractor.update_on_label(features, path, "malicious")
            self._post_threat(res, path, sha256)
            self.scan_count += 1
            res.scan_time_ms = int((time.time() - t0) * 1000)
            return res

        # Stage 3+4: Full adaptive ensemble (rules + NB + NN + meta)
        prob, details   = self.classifier.predict(vector, features=features)
        res.confidence  = prob
        res.uncertainty = details.get("nn_uncertainty", 0.0)
        res.ml_details  = details
        res.verdict     = "malicious" if prob > 0.5 else "benign"
        res.detection_method = self._format_methods(details)

        if   prob >= 0.85: res.risk_level = "CRITICAL"
        elif prob >= 0.65: res.risk_level = "HIGH"
        elif prob >= 0.40: res.risk_level = "MEDIUM"
        elif prob >= 0.20: res.risk_level = "LOW"
        else:              res.risk_level = "CLEAN"
        if res.verdict == "benign" and prob < 0.2:
            res.risk_level = "CLEAN"

        # Stage 5: 400M model (optional)
        if self._use_large_model:
            lp = self._large_model_predict(path, features)
            if lp is not None:
                prob = 0.70 * prob + 0.30 * lp
                res.confidence = prob
                res.detection_method.append(f"LM400M:{lp:.2f}")
                res.verdict = "malicious" if prob > 0.5 else "benign"

        # Adaptive self-learning — only when confident AND low uncertainty
        fired       = details.get("fired_rules", [])
        uncertainty = details.get("nn_uncertainty", 1.0)

        if prob > 0.82 and uncertainty < 0.25:
            self.classifier.learn(vector, "malicious",
                                  features=features, fired_rules=fired,
                                  weight=prob)
            self.extractor.update_on_label(features, path, "malicious")
            self.intel.add_malicious(sha256, res.threat_name or "AutoDetect")
            res.threat_name = res.threat_name or (
                "Auto." + "|".join(fired[:2]) if fired else "Auto.MLDetect"
            )
        elif prob < 0.12 and uncertainty < 0.20:
            self.classifier.learn(vector, "benign",
                                  features=features, fired_rules=fired,
                                  weight=(1.0 - prob))
            self.extractor.update_on_label(features, path, "benign")
            self.intel.add_benign(sha256)
        # else: uncertain zone — do NOT auto-label; wait for user/signature confirmation

        if res.verdict == "malicious":
            self._post_threat(res, path, sha256)

        self.scan_count += 1
        res.scan_time_ms = int((time.time() - t0) * 1000)
        self._save_state()
        return res

    def _format_methods(self, details: dict) -> list:
        methods = []
        fired = details.get("fired_rules", [])
        if fired:
            methods.append("Rules:" + "+".join(fired[:3]))
        nb  = details.get("nb_prob", 0)
        nn  = details.get("nn_prob", 0)
        std = details.get("nn_uncertainty", 0)
        methods.append(f"NB:{nb:.2f} NN:{nn:.2f}±{std:.2f}")
        w = details.get("ensemble_weights", {})
        if w:
            dom = max(w, key=w.get)
            methods.append(f"Meta:{dom}({w[dom]:.0%})")
        return methods

    def _post_threat(self, res: ScanResult, path: str, sha256: str):
        self.threat_count += 1
        action = self.quarantine.handle_threat(
            path=path, sha256=sha256,
            threat_name=res.threat_name or res.risk_level,
            risk_level=res.risk_level, confidence=res.confidence,
            detection_methods=res.detection_method,
        )
        res.quarantine_action = action

    def _large_model_predict(self, path: str, features: dict) -> Optional[float]:
        if self._large_model is None:
            cprint(C.DIM, "  [Loading 400M model...]")
            from adaptiveav.model import TransformerAVModel
            self._large_model = TransformerAVModel(load=True)
        try:
            with open(path, "rb") as f:
                data = f.read(65536)
            r = self._large_model.predict(data, feature_vector=self.extractor.to_vector(features))
            return r["malicious_prob"]
        except Exception:
            return None

    # ── Teaching ─────────────────────────────────────────────────

    def teach(self, path: str, label: str) -> bool:
        if label not in ("malicious", "benign"):
            return False
        f = self.extractor.extract(path)
        if not f.get("readable"):
            return False
        vector = self.extractor.to_vector(f)
        _, details = self.classifier.predict(vector, features=f)
        # User labels: weight=10, overrides weak auto-learned priors
        self.classifier.learn(vector, label, features=f,
                               fired_rules=details.get("fired_rules", []),
                               weight=10.0)
        # let extractor adapt too
        self.extractor.update_on_label(f, path, label)
        if label == "malicious":
            self.intel.add_malicious(f["sha256"], "UserLabeled")
        else:
            self.intel.add_benign(f["sha256"])
        self._save_state()
        return True

    def protect_app(self, path: str) -> bool:
        try:
            sha256 = hashlib.sha256(open(path, "rb").read(1024 * 1024)).hexdigest()
        except Exception:
            sha256 = ""
        self.quarantine.app_registry.register(path, sha256, "user-protected")
        self._save_state()
        return True

    # ── Directory / browser scan ──────────────────────────────────

    def scan_directory(self, dir_path: str, show: bool = True) -> list:
        results = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".")
                       and not str(Path(root) / d).startswith(str(ADAPTIVEAV_DIR))]
            for fname in files:
                r = self.scan(os.path.join(root, fname))
                results.append(r)
                if show:
                    self._print_result(r)
        return results

    def scan_browser(self) -> list:
        return self.browser_scanner.scan_all()

    # ── Output helpers ────────────────────────────────────────────

    def _print_result(self, r: ScanResult):
        color  = r.risk_color()
        icon   = "🔴" if r.verdict == "malicious" else ("⚠️ " if r.risk_level in ("MEDIUM", "LOW") else "✅")
        name   = Path(r.path).name
        conf   = f"{r.confidence*100:.0f}%"
        unc    = f"±{r.uncertainty*100:.0f}%" if r.uncertainty > 0.05 else "     "
        qa     = f" → [{r.quarantine_action['action'].upper()}]" if r.quarantine_action else ""
        threat = f" [{r.threat_name}]" if r.threat_name else ""
        print(f"  {icon} {color}{r.risk_level:8}{C.RESET} {name:<34} {conf}{unc}{qa}{threat}")

    def print_browser_result(self, br: BrowserScanResult):
        if br.threat_count == 0:
            cprint(C.GREEN, f"  ✅ [{br.browser}/{br.profile}] Clean ({br.total_exts_checked} exts, {br.total_urls_checked} URLs)")
        else:
            cprint(C.YELLOW, f"  ⚠️  [{br.browser}/{br.profile}] {br.threat_count} threats:")
            for t in br.extension_threats[:3]:
                cprint(C.RED, f"      Extension '{t['name']}': {t['verdict']}")
            for t in br.url_threats[:3]:
                cprint(C.YELLOW, f"      URL: {t['domain']} — {t['reasons'][0]}")
            for t in br.js_threats[:3]:
                cprint(C.RED, f"      JS: {t['pattern']}")

    def report(self):
        qs  = self.quarantine.stats()
        cl  = self.classifier

        cprint(C.CYAN + C.BOLD,
               "\n╔══ AdaptiveAV v3 — Truly Adaptive Report ═══════════════════╗")
        print(f"  Version          : {self.VERSION}")
        print(f"  Platform         : {platform.system()} {platform.machine()}")
        print(f"  Total Scans      : {self.scan_count}")
        print(f"  Threats Found    : {self.threat_count}")

        cprint(C.CYAN, "\n  ── Adaptive ML Engine ──────────────────────────────────────")
        print(f"  Samples Learned  : {cl.n_learned}")
        print(f"  Replay Buffer    : {len(cl.replay)} items  (cap 2000, prioritized)")
        print(f"  Drift Detections : {cl.n_drifts}  (Page-Hinkley, auto-replay on drift)")
        print(f"  Recent Accuracy  : {cl.recent_accuracy()*100:.1f}%  (last {len(cl._recent_accs)})")
        print(f"  NN Trained Steps : {cl.nn.n_trained}  (Adam optimizer, full backprop)")
        print(f"  NB Trained Steps : {cl.nb.n_trained}  (Welford + exponential forgetting)")
        print(f"  Current LR       : {cl.nn.lr:.6f}")
        print(f"  Drift Stability  : {cl.drift.stability()*100:.1f}%")
        print(f"  Meta-Updates     : {cl.meta.n_updates}  (per-cluster ensemble weights)")

        cprint(C.CYAN, "\n  ── Top Features (by learned gradient importance) ────────────")
        for fname, imp in cl.top_features():
            bar = "█" * max(1, int(imp * 25))
            print(f"    {fname:<30} {bar} {imp:.4f}")

        cprint(C.CYAN, "\n  ── Adaptive Rule Weights (credit-assigned) ──────────────────")
        accs = cl.rules.accuracy_per_rule()
        for rule, w in sorted(cl.rules._weights.items(), key=lambda t: t[1], reverse=True)[:7]:
            acc_r  = accs.get(rule, 0.0)
            fires  = cl.rules._fires.get(rule, 0)
            bar    = "▓" * max(1, int(w / 15.0 * 20))
            print(f"    {rule:<28} {bar:<22} w={w:5.2f}  acc={acc_r*100:3.0f}%  fires={fires}")

        cprint(C.CYAN, "\n  ── Quarantine ────────────────────────────────────────────────")
        print(f"  Quarantined      : {qs['total_quarantined']}")
        print(f"  Watchlisted      : {qs['watchlist_count']}")
        print(f"  Protected Apps   : {qs['protected_apps']}")
        print(f"  Sandbox Avail    : {self.quarantine.sandbox_mgr.is_sandboxable()}")
        print(f"  400M Model       : {'enabled' if self._use_large_model else 'off (--large to enable)'}")

        top = self.intel.families.most_common(5)
        if top:
            cprint(C.YELLOW, "\n  ── Top Threat Families ──────────────────────────────────────")
            for name, count in top:
                print(f"    • {name}: {count}")

        cprint(C.CYAN + C.BOLD,
               "╚═════════════════════════════════════════════════════════════╝\n")

    # ── Bootstrap ────────────────────────────────────────────────

    def _bootstrap(self):
        rng = random.Random(42)

        def mal():
            return [rng.uniform(0.8,1.0), 1, rng.uniform(0.7,1.0), rng.uniform(0.5,1.0),
                    rng.uniform(0,0.1), rng.uniform(0.2,0.5), rng.uniform(0.3,0.7), rng.uniform(0.5,1.0),
                    0.3, rng.uniform(0.5,1.0), 1.0, rng.uniform(0.3,1.0), rng.uniform(0.2,0.8),
                    rng.uniform(0.3,0.8), rng.uniform(0.5,1.0), rng.uniform(0.2,0.8), rng.uniform(0.2,0.8),
                    rng.uniform(0.3,1.0), rng.uniform(0.7,1.0), rng.uniform(0.1,0.9)]
        def ben():
            return [rng.uniform(0.3,0.7), 0, 0, rng.uniform(0.3,0.6), rng.uniform(0,0.05),
                    rng.uniform(0.6,0.95), rng.uniform(0,0.2), rng.uniform(0,0.3), 0.1,
                    0, 0, rng.uniform(0,0.1), rng.uniform(0,0.1), rng.uniform(0,0.1),
                    0, 0, 0, rng.uniform(0,0.1), rng.uniform(0.2,0.5), rng.uniform(0,0.5)]

        for _ in range(30): self.classifier.learn(mal(), "malicious", weight=0.5)
        for _ in range(30): self.classifier.learn(ben(), "benign", weight=0.5)

    # ── Persistence ──────────────────────────────────────────────

    def _save_state(self):
        try:
            state = {
                "classifier": self.classifier.state_dict(),
                "extractor": self.extractor.state_dict(),
                "intel_mal":  self.intel.malicious,
                "intel_ben":  list(self.intel.benign),
                "families":   dict(self.intel.families),
                "scans":      self.scan_count,
                "threats":    self.threat_count,
                "version":    self.VERSION,
            }
            MODEL_CACHE.parent.mkdir(exist_ok=True, parents=True)
            with open(MODEL_CACHE, "wb") as f:
                pickle.dump(state, f)
        except Exception:
            pass

    def _load_state(self):
        if not MODEL_CACHE.exists():
            return
        try:
            with open(MODEL_CACHE, "rb") as f:
                s = pickle.load(f)
            self.classifier.load_state(s.get("classifier", {}))
            if "extractor" in s:
                self.extractor.load_state(s.get("extractor", {}))
            self.intel.malicious = s.get("intel_mal", {})
            self.intel.benign    = set(s.get("intel_ben", []))
            self.intel.families  = Counter(s.get("families", {}))
            self.scan_count      = s.get("scans", 0)
            self.threat_count    = s.get("threats", 0)
            cprint(C.DIM,
                   f"  [v3 loaded: {self.classifier.n_learned} learned, "
                   f"{self.scan_count} scans, "
                   f"{len(self.intel.malicious)} threats, "
                   f"{self.classifier.n_drifts} drifts]")
        except Exception:
            pass