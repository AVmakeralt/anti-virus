// popup.js — AdaptiveAV Shield popup
"use strict";

function renderAlerts(alerts) {
  const list = document.getElementById("alerts-list");
  if (!alerts || alerts.length === 0) {
    list.innerHTML = '<div class="empty">No threats detected today ✓</div>';
    return;
  }
  const recent = alerts.slice(-5).reverse();
  list.innerHTML = recent.map(a => {
    let hostname = a.url;
    try { hostname = new URL(a.url).hostname; } catch {}
    const threat = a.threats?.[0]?.detail || a.threats?.[0]?.type || "Unknown";
    return `
      <div class="alert-item ${a.risk}">
        <div><span class="alert-risk">${a.risk}</span> — ${threat}</div>
        <div class="alert-url">${hostname}</div>
      </div>`;
  }).join("");
}

async function refresh() {
  const resp = await chrome.runtime.sendMessage({ type: "get_state" });
  if (!resp?.state) return;
  const { state } = resp;

  document.getElementById("scanned").textContent = state.scannedCount || 0;
  document.getElementById("blocked").textContent = state.blockedCount || 0;
  document.getElementById("threats").textContent = (state.alertsToday || []).length;

  renderAlerts(state.alertsToday || []);

  // Settings toggles
  const settings = state.settings || {};
  for (const [key, val] of Object.entries(settings)) {
    const el = document.getElementById(key);
    if (el) el.checked = !!val;
  }
}

// Toggle handlers
["blockMalicious", "scanDownloads", "scanJavaScript", "showNotifications"].forEach(id => {
  document.getElementById(id).addEventListener("change", async (e) => {
    await chrome.runtime.sendMessage({
      type: "update_settings",
      settings: { [id]: e.target.checked }
    });
  });
});

document.getElementById("clear-btn").addEventListener("click", async () => {
  await chrome.storage.local.set({ blockedCount: 0, scannedCount: 0 });
  location.reload();
});

document.getElementById("reload-btn").addEventListener("click", refresh);

// Initial load
refresh();