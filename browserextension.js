/**
 * AdaptiveAV Browser Shield — Background Service Worker
 * =======================================================
 * ALL threat detection happens locally in the browser.
 * NO external API calls. NO data leaves the machine.
 *
 * Features:
 *  - Real-time URL reputation check (local rules)
 *  - Download file threat detection
 *  - Phishing/typosquatting detection
 *  - Malicious redirect detection
 *  - Suspicious JS pattern detection via content scripts
 *  - Communicates with local AdaptiveAV daemon (if running) via native messaging
 */

"use strict";

// ── Local Threat Intelligence (embedded, no network) ─────────────

const MALICIOUS_DOMAINS = new Set([
  // Known malware distribution domains (static list, updated via extension updates)
  "malware-download.net", "free-crack.xyz", "virus-test.tk",
  "phishing-example.ml", "fake-paypal.ga",
]);

const PHISHING_KEYWORDS = [
  "login", "signin", "account", "verify", "secure", "update",
  "confirm", "banking", "paypal", "amazon", "apple", "microsoft",
  "google", "facebook", "netflix", "steam",
];

const SUSPICIOUS_TLDS = new Set([".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".download"]);

const TYPOSQUAT_BRANDS = [
  { real: "paypal.com",    typos: ["paypa1.com", "paypai.com", "paypa1.net", "paypol.com"] },
  { real: "google.com",    typos: ["g00gle.com", "googIe.com", "go0gle.com", "gooogle.com"] },
  { real: "amazon.com",    typos: ["amaz0n.com", "amazoon.com", "arnazon.com"] },
  { real: "microsoft.com", typos: ["micosoft.com", "microsft.com", "micros0ft.com"] },
  { real: "apple.com",     typos: ["app1e.com", "appIe.com", "aple.com"] },
  { real: "bankofamerica.com", typos: ["bankofamerica-secure.com", "bofa-login.com"] },
];

const MALICIOUS_URL_PATTERNS = [
  /\.(exe|dll|bat|cmd|ps1|vbs|scr|hta|jar|msi|dmg)(\?|$)/i,
  /eval\s*\(\s*atob\s*\(/i,
  /base64_decode\s*\(/i,
  /\/shell\.php/i,
  /\/c99\.php/i,
  /(data:text\/html;base64)/i,
  /javascript:void\(0\)\s*;?\s*eval/i,
];

const DANGEROUS_DOWNLOAD_EXTENSIONS = new Set([
  ".exe", ".dll", ".bat", ".cmd", ".ps1", ".vbs", ".js",
  ".jar", ".msi", ".scr", ".hta", ".pif", ".com",
  ".sh", ".run", ".elf", ".deb", ".rpm",
]);

// ── State ─────────────────────────────────────────────────────────

const state = {
  blockedCount:  0,
  scannedCount:  0,
  alertsToday:   [],
  lastScan:      null,
  settings: {
    blockMalicious:   true,
    warnSuspicious:   true,
    scanDownloads:    true,
    scanJavaScript:   true,
    showNotifications: true,
  }
};

// Load settings from storage
chrome.storage.local.get(["settings", "blockedCount", "scannedCount"], (data) => {
  if (data.settings)      Object.assign(state.settings, data.settings);
  if (data.blockedCount)  state.blockedCount = data.blockedCount;
  if (data.scannedCount)  state.scannedCount = data.scannedCount;
});

// ── URL Analysis ─────────────────────────────────────────────────

function analyzeURL(url) {
  const result = {
    url,
    threats:   [],
    riskLevel: "CLEAN",
    block:     false,
  };

  let urlObj;
  try {
    urlObj = new URL(url);
  } catch {
    return result;
  }

  const hostname = urlObj.hostname.toLowerCase();
  const fullUrl  = url.toLowerCase();
  const tld      = hostname.match(/\.[a-z]{2,}$/)?.[0] || "";

  // 1. Known malicious domains
  if (MALICIOUS_DOMAINS.has(hostname)) {
    result.threats.push({ type: "known_malicious", detail: hostname });
    result.riskLevel = "CRITICAL";
    result.block = true;
  }

  // 2. Typosquatting detection
  for (const brand of TYPOSQUAT_BRANDS) {
    if (brand.typos.some(t => hostname.includes(t.replace(".com", "")))) {
      result.threats.push({ type: "typosquatting", detail: `Possible impersonation of ${brand.real}` });
      result.riskLevel = "HIGH";
      result.block = state.settings.blockMalicious;
    }
  }

  // 3. Suspicious TLD
  if (SUSPICIOUS_TLDS.has(tld)) {
    result.threats.push({ type: "suspicious_tld", detail: `High-risk TLD: ${tld}` });
    result.riskLevel = result.riskLevel === "CLEAN" ? "MEDIUM" : result.riskLevel;
  }

  // 4. Malicious URL patterns
  for (const pat of MALICIOUS_URL_PATTERNS) {
    if (pat.test(url)) {
      result.threats.push({ type: "url_pattern", detail: pat.source.slice(0, 60) });
      result.riskLevel = "HIGH";
    }
  }

  // 5. Phishing — brand keyword in non-brand domain
  const domainParts = hostname.split(".");
  const baseDomain  = domainParts.slice(-2).join(".");
  for (const kw of PHISHING_KEYWORDS) {
    if (hostname.includes(kw) && !hostname.endsWith(kw + ".com")) {
      // Subdomain contains brand keyword but isn't the real domain
      if (urlObj.pathname.length > 1 || urlObj.search.length > 1) {
        result.threats.push({ type: "phishing_keyword", detail: `Brand keyword "${kw}" in suspicious position` });
        result.riskLevel = result.riskLevel === "CLEAN" ? "MEDIUM" : result.riskLevel;
        break;
      }
    }
  }

  // 6. IP-based URL with executable
  if (/^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname)) {
    if (MALICIOUS_URL_PATTERNS[0].test(url)) {
      result.threats.push({ type: "ip_executable", detail: "Executable served from IP address" });
      result.riskLevel = "CRITICAL";
      result.block = true;
    }
  }

  // 7. Excessive subdomains (DGA-like)
  if (domainParts.length > 5) {
    result.threats.push({ type: "deep_subdomain", detail: `${domainParts.length} subdomain levels` });
    result.riskLevel = result.riskLevel === "CLEAN" ? "LOW" : result.riskLevel;
  }

  return result;
}

// ── Download Scanner ──────────────────────────────────────────────

async function scanDownload(downloadItem) {
  const filename = downloadItem.filename || downloadItem.url || "";
  const ext = filename.match(/\.[^.]+$/)?.[0]?.toLowerCase() || "";
  const result = {
    downloadId: downloadItem.id,
    filename,
    threats: [],
    riskLevel: "CLEAN",
    pause: false,
  };

  // Dangerous extension
  if (DANGEROUS_DOWNLOAD_EXTENSIONS.has(ext)) {
    // Check URL too
    const urlAnalysis = analyzeURL(downloadItem.url || "");
    result.threats.push({
      type:   "dangerous_extension",
      detail: `Executable file type: ${ext}`,
    });
    result.riskLevel = "MEDIUM";

    if (urlAnalysis.threats.length > 0) {
      result.threats.push(...urlAnalysis.threats);
      result.riskLevel = "HIGH";
      result.pause = true;
    }
  }

  // URL analysis for any download
  const urlResult = analyzeURL(downloadItem.url || "");
  if (urlResult.riskLevel !== "CLEAN") {
    result.threats.push(...urlResult.threats);
    result.riskLevel = urlResult.riskLevel;
    result.pause = result.pause || urlResult.block;
  }

  return result;
}

// ── Event Listeners ───────────────────────────────────────────────

// Block/warn on navigation
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  if (details.frameId !== 0) return;  // Main frame only

  const analysis = analyzeURL(details.url);
  state.scannedCount++;

  if (analysis.riskLevel === "CLEAN") return;

  state.alertsToday.push({
    timestamp: Date.now(),
    url:       details.url,
    risk:      analysis.riskLevel,
    threats:   analysis.threats,
  });

  // Update badge
  updateBadge(analysis.riskLevel);

  if (analysis.block && state.settings.blockMalicious) {
    state.blockedCount++;
    // Redirect to block page
    chrome.tabs.update(details.tabId, {
      url: chrome.runtime.getURL(`blocked.html?url=${encodeURIComponent(details.url)}&risk=${analysis.riskLevel}&threats=${encodeURIComponent(JSON.stringify(analysis.threats))}`)
    });
  } else if (state.settings.warnSuspicious) {
    showNotification("Suspicious Site", `${analysis.riskLevel} risk: ${new URL(details.url).hostname}`, analysis.riskLevel);
  }

  saveState();
});

// Scan downloads
chrome.downloads.onCreated.addListener(async (downloadItem) => {
  if (!state.settings.scanDownloads) return;

  const result = await scanDownload(downloadItem);
  if (result.riskLevel === "CLEAN") return;

  if (result.pause) {
    chrome.downloads.pause(downloadItem.id);
    showNotification(
      "⚠ Download Blocked",
      `Suspicious download paused: ${downloadItem.filename.split("/").pop()}\nRisk: ${result.riskLevel}`,
      result.riskLevel
    );
  } else {
    showNotification(
      "Download Warning",
      `Potentially dangerous file: ${downloadItem.filename.split("/").pop()}`,
      result.riskLevel
    );
  }
});

// Listen for JS pattern reports from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "js_threat") {
    state.alertsToday.push({
      timestamp: Date.now(),
      url:       sender.url,
      risk:      "HIGH",
      threats:   [{ type: "js_pattern", detail: message.pattern }],
    });
    updateBadge("HIGH");
    if (state.settings.showNotifications) {
      showNotification("Malicious JavaScript", `Detected on ${new URL(sender.url || "about:blank").hostname}: ${message.pattern}`, "HIGH");
    }
  }

  if (message.type === "get_state") {
    sendResponse({ state });
    return true;
  }

  if (message.type === "update_settings") {
    Object.assign(state.settings, message.settings);
    chrome.storage.local.set({ settings: state.settings });
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === "resume_download") {
    chrome.downloads.resume(message.downloadId);
    sendResponse({ ok: true });
    return true;
  }
});

// ── Helpers ───────────────────────────────────────────────────────

function updateBadge(riskLevel) {
  const colors = {
    CRITICAL: "#ff0000",
    HIGH:     "#ff6600",
    MEDIUM:   "#ffaa00",
    LOW:      "#ffdd00",
    CLEAN:    "#00aa00",
  };
  chrome.action.setBadgeBackgroundColor({ color: colors[riskLevel] || "#999" });
  chrome.action.setBadgeText({ text: riskLevel === "CLEAN" ? "" : "!" });
}

function showNotification(title, message, riskLevel) {
  if (!state.settings.showNotifications) return;
  chrome.notifications.create({
    type:     "basic",
    iconUrl:  "icons/icon48.png",
    title:    `AdaptiveAV: ${title}`,
    message:  message,
    priority: riskLevel === "CRITICAL" ? 2 : 1,
  });
}

function saveState() {
  chrome.storage.local.set({
    blockedCount: state.blockedCount,
    scannedCount: state.scannedCount,
  });
}

// Periodic cleanup of old alerts (keep last 24h)
setInterval(() => {
  const cutoff = Date.now() - 86400 * 1000;
  state.alertsToday = state.alertsToday.filter(a => a.timestamp > cutoff);
}, 60 * 60 * 1000);