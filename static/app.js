/** Относительные пути — тот же хост/порт, что у открытой страницы (без сборки origin). */
function apiPath(p) {
  return p.startsWith("/") ? p : "/" + p;
}

/** Несколько попыток при кратковременном «Failed to fetch» (Windows / сон / антивирус). */
async function fetchWithRetry(url, options, tries) {
  const n = tries != null ? tries : 4;
  let last;
  for (let i = 0; i < n; i++) {
    try {
      return await fetch(url, options);
    } catch (e) {
      last = e;
      if (i + 1 < n) await new Promise((r) => setTimeout(r, 350 * (i + 1)));
    }
  }
  throw last;
}

async function loadConfig() {
  const r = await fetchWithRetry(apiPath("/api/config"), { cache: "no-store", credentials: "same-origin" });
  if (!r.ok) throw new Error("config HTTP " + r.status);
  return r.json();
}

function fetchHintLines() {
  return [
    "",
    "Нет связи с сервером: запустите python web_server.py и откройте http://127.0.0.1:8765 в обычном браузере (не file://).",
    "Проверка: curl http://127.0.0.1:8765/api/ping",
  ];
}

function parseJsonResponse(text, status) {
  if (!text || !text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("Ответ сервера не JSON (HTTP " + status + "): " + text.slice(0, 600));
  }
}

/** Человекочитаемое время для бейджа (избегаем «1022 с»). */
function formatRunElapsed(sec) {
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return r + " с";
  return m + " мин " + r + " с";
}

/** POST /api/run теперь 202 + фон; опрос до готовности. */
async function waitForRunJob(jobId, badgeEl, tStart) {
  const t0 = tStart != null ? tStart : Date.now();
  const deadline = Date.now() + 90 * 60 * 1000;
  while (Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 1200));
    const r2 = await fetchWithRetry(apiPath("/api/job/" + encodeURIComponent(jobId)), {
      cache: "no-store",
      credentials: "same-origin",
    });
    const t2 = await r2.text();
    const st = parseJsonResponse(t2, r2.status);
    if (!r2.ok) {
      throw new Error((st && st.error) || "статус задачи HTTP " + r2.status);
    }
    if (st.status === "running") {
      const sec = Math.floor((Date.now() - t0) / 1000);
      const n = (st.logs && st.logs.length) || 0;
      badgeEl.textContent =
        "идёт… " + formatRunElapsed(sec) + " — фон, лог " + n + " строк · типично 5–25 мин";
      continue;
    }
    if (st.status === "error") {
      throw new Error(st.error || "ошибка сверки");
    }
    if (st.status === "done") {
      const out = { ...st };
      delete out.status;
      return out;
    }
    throw new Error("неизвестный статус задачи");
  }
  throw new Error("Превышено время ожидания сверки (90 мин).");
}

function setPill(text, ok) {
  const el = document.getElementById("connPill");
  el.textContent = text;
  el.style.borderColor = ok ? "var(--accent-dim)" : "var(--border)";
  el.style.color = ok ? "var(--accent)" : "var(--muted)";
}

function appendLog(text) {
  const box = document.getElementById("logBox");
  box.textContent += text + "\n";
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatInlineBoldAfterEscape(e) {
  return e.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

/** Если ответ — только маркированный список, показываем как <ul> вместо превью diff-карты. */
function renderGemmaMarkdown(md) {
  const raw = (md || "").trim();
  if (!raw) {
    return '<p class="muted">Отчёт отключён или модель не ответила.</p>';
  }
  if (raw.startsWith("[Gemma") || raw.startsWith("[Ollama")) {
    return '<pre class="md-error">' + escapeHtml(raw) + "</pre>";
  }
  const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  const bullet = /^[-*]\s+/;
  if (lines.length >= 1 && lines.every((l) => bullet.test(l))) {
    const items = lines.map((l) => l.replace(bullet, ""));
    return (
      '<ul class="fix-list">' +
      items.map((i) => "<li>" + formatInlineBoldAfterEscape(escapeHtml(i)) + "</li>").join("") +
      "</ul>"
    );
  }
  return typeof marked !== "undefined" ? marked.parse(raw) : "<pre>" + escapeHtml(raw) + "</pre>";
}

document.addEventListener("DOMContentLoaded", async () => {
  let ollamaReadSec = 600;

  if (window.location.protocol === "file:") {
    setPill("нужен http://127.0.0.1:8765", false);
    const box = document.getElementById("logBox");
    if (box) {
      box.textContent =
        "Страница открыта как файл (file://). Запросы к API с диска браузером блокируются — поэтому «Failed to fetch».\n\n" +
        "1) В терминале: python web_server.py\n" +
        "2) В браузере: http://127.0.0.1:8765\n";
    }
    document.getElementById("btnRun").addEventListener("click", () => {
      appendLog("Откройте панель через сервер (http://127.0.0.1:8765), а не двойным щелчком по HTML.");
    });
    return;
  }

  try {
    const c = await loadConfig();
    ollamaReadSec = Number(c.ollama_timeout_read) || 600;
    document.getElementById("urlSite").value = c.url_site || "";
    document.getElementById("figmaKey").value = c.figma_file_key || "";
    document.getElementById("figmaNode").value = c.figma_node_id || "";
    const figmaHint = c.figma_url_hint || "";
    const figmaUrlEl = document.getElementById("figmaUrl");
    if (figmaHint) {
      figmaUrlEl.value = figmaHint;
    }
    document.getElementById("figmaUrlHint").textContent = figmaHint
      ? "Пример из config.json: " + figmaHint
      : "Укажите в config.json figma.file_key и figma.node_id — сюда подставится пример ссылки.";
    const us = (c.url_site || "").trim();
    document.getElementById("urlSiteHint").textContent = us
      ? "Пример из config.json: " + us
      : "Вставьте URL опубликованной страницы или локального сервера (например http://127.0.0.1:8080).";
    document.getElementById("figScale").value = c.figma_scale || 1;
    document.getElementById("winW").value = c.window_w;
    document.getElementById("winH").value = c.window_h;
    document.getElementById("capWait").value =
      c.capture_wait_seconds != null ? c.capture_wait_seconds : 12;
    document.getElementById("thr").value = c.diff_threshold_pct;
    document.getElementById("pixThr").value = c.pixel_threshold;
    document.getElementById("shift").value = c.tolerance_shift_px;
    document.getElementById("speck").value = c.tolerance_speckle_iter;
    setPill("сервер OK", true);
  } catch (e) {
    setPill("ошибка config.json", false);
    const box = document.getElementById("logBox");
    const msg = e && e.message ? e.message : String(e);
    if (box) {
      box.textContent =
        "Сервер отвечает, но /api/config не удалось разобрать: " +
        msg +
        "\n\nПроверьте JSON в config.json (запятые, кавычки). Запрос: GET /api/config\n";
      fetchHintLines().forEach((line) => {
        box.textContent += line + "\n";
      });
    }
  }

  document.getElementById("btnRun").addEventListener("click", async () => {
    const btn = document.getElementById("btnRun");
    const badge = document.getElementById("statusBadge");
    const gemmaMd = document.getElementById("gemmaMd");
    document.getElementById("logBox").textContent = "";
    gemmaMd.innerHTML = "";

    btn.disabled = true;
    badge.className = "status run";
    badge.textContent = "идёт…";

    const t0 = Date.now();
    let tick = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      badge.textContent =
        "идёт… " +
        formatRunElapsed(s) +
        " — ждём /api/run; >1 мин — Ctrl+F5 + web_server.py. Ollama до ~" +
        Math.round(ollamaReadSec) +
        " с на один HTTP к модели (не весь прогон)";
    }, 1000);

    const body = {
      url_site: document.getElementById("urlSite").value.trim(),
      figma_url: document.getElementById("figmaUrl").value.trim(),
      figma_file_key: document.getElementById("figmaKey").value.trim(),
      figma_node_id: document.getElementById("figmaNode").value.trim(),
      figma_scale: parseInt(document.getElementById("figScale").value, 10),
      window_w: parseInt(document.getElementById("winW").value, 10),
      window_h: parseInt(document.getElementById("winH").value, 10),
      diff_threshold_pct: parseFloat(String(document.getElementById("thr").value).replace(",", ".")),
      pixel_threshold: parseInt(document.getElementById("pixThr").value, 10),
      tolerance_shift_px: parseInt(document.getElementById("shift").value, 10),
      tolerance_speckle_iter: parseInt(document.getElementById("speck").value, 10),
      capture_wait_seconds: parseFloat(
        String(document.getElementById("capWait").value).replace(",", ".")
      ),
      figma_refresh: document.getElementById("figmaRefresh").checked,
      use_gemma: document.getElementById("useGemma").checked,
      use_model: document.getElementById("useModel").checked,
      gemma_use_image: document.getElementById("gemmaImg").checked,
    };

    try {
      const r = await fetchWithRetry(apiPath("/api/run"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        cache: "no-store",
        credentials: "same-origin",
      });
      if (tick) {
        clearInterval(tick);
        tick = null;
      }
      const text = await r.text();
      let data;
      try {
        data = parseJsonResponse(text, r.status);
      } catch (parseErr) {
        appendLog(String(parseErr));
        fetchHintLines().forEach((line) => appendLog(line));
        badge.className = "status fail";
        badge.textContent = "ошибка";
        return;
      }
      if (!r.ok) {
        appendLog(data.error || "ошибка");
        (data.logs || []).forEach((l) => appendLog(l));
        badge.className = "status fail";
        badge.textContent = "ошибка";
        return;
      }
      if (r.status === 202 && data.job_id) {
        appendLog(data.message || "Сверка в фоне, ждём…");
        data = await waitForRunJob(data.job_id, badge, t0);
      }
      (data.logs || []).forEach((l) => appendLog(l));
      appendLog("");
      appendLog("Отчёт: " + data.report_txt);
      if (data.report_html) appendLog("HTML: " + data.report_html);
      appendLog("Артефакты: " + data.witness_dir);
      appendLog("MSE: " + data.mse + " | пиксели: " + data.changed_ratio_pct + "%");
      if (data.model_prob_fail != null) {
        appendLog("CNN P(fail): " + data.model_prob_fail);
      }

      badge.className = "status " + (data.ok ? "pass" : "fail");
      badge.textContent = data.ok ? "PASS" : "FAIL";

      const md = data.gemma_markdown || "";
      gemmaMd.innerHTML = renderGemmaMarkdown(md);
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      appendLog(msg);
      if (msg === "Failed to fetch" || (e instanceof TypeError && String(msg).toLowerCase().includes("fetch"))) {
        fetchHintLines().forEach((line) => appendLog(line));
      }
      badge.className = "status fail";
      badge.textContent = "сеть";
    } finally {
      if (tick) clearInterval(tick);
      btn.disabled = false;
    }
  });
});
