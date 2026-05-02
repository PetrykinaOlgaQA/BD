async function loadConfig() {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error("config");
  return r.json();
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

document.addEventListener("DOMContentLoaded", async () => {
  try {
    const c = await loadConfig();
    document.getElementById("urlSite").value = c.url_site || "";
    document.getElementById("figmaKey").value = c.figma_file_key || "";
    document.getElementById("figmaNode").value = c.figma_node_id || "";
    document.getElementById("figScale").value = c.figma_scale || 2;
    document.getElementById("winW").value = c.window_w;
    document.getElementById("winH").value = c.window_h;
    document.getElementById("thr").value = c.diff_threshold_pct;
    document.getElementById("pixThr").value = c.pixel_threshold;
    document.getElementById("shift").value = c.tolerance_shift_px;
    document.getElementById("speck").value = c.tolerance_speckle_iter;
    setPill("готово", true);
  } catch {
    setPill("нет config", false);
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

    const body = {
      url_site: document.getElementById("urlSite").value.trim(),
      figma_file_key: document.getElementById("figmaKey").value.trim(),
      figma_node_id: document.getElementById("figmaNode").value.trim(),
      figma_scale: parseInt(document.getElementById("figScale").value, 10),
      window_w: parseInt(document.getElementById("winW").value, 10),
      window_h: parseInt(document.getElementById("winH").value, 10),
      diff_threshold_pct: parseFloat(String(document.getElementById("thr").value).replace(",", ".")),
      pixel_threshold: parseInt(document.getElementById("pixThr").value, 10),
      tolerance_shift_px: parseInt(document.getElementById("shift").value, 10),
      tolerance_speckle_iter: parseInt(document.getElementById("speck").value, 10),
      use_gemma: document.getElementById("useGemma").checked,
      use_model: document.getElementById("useModel").checked,
      gemma_use_image: document.getElementById("gemmaImg").checked,
    };

    try {
      const r = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) {
        appendLog(data.error || "ошибка");
        (data.logs || []).forEach((l) => appendLog(l));
        badge.className = "status fail";
        badge.textContent = "ошибка";
        return;
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
      gemmaMd.innerHTML = md
        ? (typeof marked !== "undefined" ? marked.parse(md) : "<pre>" + md + "</pre>")
        : "<p class=\"muted\">Отчёт отключён или модель не ответила.</p>";
    } catch (e) {
      appendLog(String(e));
      badge.className = "status fail";
      badge.textContent = "сеть";
    } finally {
      btn.disabled = false;
    }
  });
});
