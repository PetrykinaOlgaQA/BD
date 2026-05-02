from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import FigmaVsSiteConfig, run_figma_vs_site


def load_cfg(path: str) -> dict:
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    with open(os.path.join(ROOT, "config.example.json"), encoding="utf-8") as f:
        return json.load(f)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Сайт vs макет Figma")
        self.geometry("820x640")
        self.cfg_path = os.path.join(ROOT, "config.json")
        self._q: queue.Queue = queue.Queue()
        self._busy = False

        f = tk.Frame(self, padx=12, pady=10)
        f.pack(fill=tk.BOTH, expand=True)
        r = 0
        tk.Label(f, text="URL страницы под тестом (локальный dev-сервер)", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w"
        )
        r += 1
        self.url_site = tk.Entry(f, width=92)
        self.url_site.grid(row=r, column=0, columnspan=3, sticky="we", pady=(0, 8))
        r += 1
        tk.Label(f, text="Токен Figma: переменная окружения FIGMA_ACCESS_TOKEN", fg="gray").grid(
            row=r, column=0, columnspan=3, sticky="w"
        )
        r += 1

        tk.Label(f, text="Окно W×H").grid(row=r, column=0, sticky="w")
        wf = tk.Frame(f)
        wf.grid(row=r, column=1, sticky="w")
        self.win_w = tk.Entry(wf, width=7)
        self.win_w.pack(side=tk.LEFT)
        tk.Label(wf, text=" × ").pack(side=tk.LEFT)
        self.win_h = tk.Entry(wf, width=7)
        self.win_h.pack(side=tk.LEFT)
        r += 1
        tk.Label(f, text="Порог отличий %").grid(row=r, column=0, sticky="w")
        self.thr = tk.Entry(f, width=10)
        self.thr.grid(row=r, column=1, sticky="w")
        r += 1
        tk.Label(f, text="Сдвиг 0–5 px").grid(row=r, column=0, sticky="w")
        self.shift = tk.Spinbox(f, from_=0, to=5, width=5)
        self.shift.grid(row=r, column=1, sticky="w")
        r += 1
        tk.Label(f, text="Opening 0–5").grid(row=r, column=0, sticky="w")
        self.speck = tk.Spinbox(f, from_=0, to=5, width=5)
        self.speck.grid(row=r, column=1, sticky="w")
        r += 1
        tk.Label(f, text="Порог яркости diff").grid(row=r, column=0, sticky="w")
        self.pix_thr = tk.Spinbox(f, from_=0, to=255, width=5)
        self.pix_thr.grid(row=r, column=1, sticky="w")
        r += 1

        self.use_gemma = tk.BooleanVar(value=True)
        self.use_model = tk.BooleanVar(value=True)
        self.gemma_img = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text="Текст багов (Gemma / Ollama)", variable=self.use_gemma).grid(row=r, column=1, sticky="w")
        r += 1
        tk.Checkbutton(f, text="CNN по diff", variable=self.use_model).grid(row=r, column=1, sticky="w")
        r += 1
        tk.Checkbutton(f, text="Передавать diff-картинку в модель", variable=self.gemma_img).grid(row=r, column=1, sticky="w")
        r += 1

        self.btn = tk.Button(f, text="Сравнить с макетом Figma", command=self.on_run, font=("", 11, "bold"))
        self.btn.grid(row=r, column=0, columnspan=3, pady=10, sticky="w")
        r += 1

        tk.Label(f, text="Лог", font=("", 9, "bold")).grid(row=r, column=0, sticky="w")
        r += 1
        self.log = scrolledtext.ScrolledText(f, height=22, state=tk.DISABLED, font=("Consolas", 10))
        self.log.grid(row=r, column=0, columnspan=3, sticky="nsew")
        f.rowconfigure(r, weight=1)
        f.columnconfigure(0, weight=1)

        self._apply_cfg_defaults()
        self.after(120, self._pump)

    def _apply_cfg_defaults(self):
        c = load_cfg(self.cfg_path)
        self.url_site.delete(0, tk.END)
        self.url_site.insert(0, c.get("url_site", c.get("url_local", "http://127.0.0.1:5173")))
        w, h = c.get("window_size", [1280, 720])
        self.win_w.delete(0, tk.END)
        self.win_w.insert(0, str(w))
        self.win_h.delete(0, tk.END)
        self.win_h.insert(0, str(h))
        self.thr.delete(0, tk.END)
        self.thr.insert(0, str(c.get("diff_threshold_pct", 0.5)))
        self.shift.delete(0, tk.END)
        self.shift.insert(0, str(int(c.get("tolerance_shift_px", 2))))
        self.speck.delete(0, tk.END)
        self.speck.insert(0, str(int(c.get("tolerance_speckle_iter", 1))))
        self.pix_thr.delete(0, tk.END)
        self.pix_thr.insert(0, str(int(c.get("pixel_threshold", 30))))

    def _append(self, s: str):
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, s + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _pump(self):
        try:
            while True:
                item = self._q.get_nowait()
                if isinstance(item, tuple) and item[0] == "ok":
                    self._busy = False
                    self.btn.configure(state=tk.NORMAL)
                    out = item[1]
                    self._append(f"Отчёт TXT: {out.report_txt}")
                    if getattr(out, "report_html", None):
                        self._append(f"Отчёт HTML: {out.report_html}")
                    self._append(f"Артефакты: {out.witness_dir}")
                    if out.gemma_text:
                        self._append("")
                        self._append(out.gemma_text)
                elif isinstance(item, tuple) and item[0] == "err":
                    self._busy = False
                    self.btn.configure(state=tk.NORMAL)
                    e = item[1]
                    self._append(str(e))
                    messagebox.showerror("", str(e))
                else:
                    self._append(str(item))
        except queue.Empty:
            pass
        self.after(120, self._pump)

    def on_run(self):
        if self._busy:
            return
        site = self.url_site.get().strip()
        if not site:
            messagebox.showwarning("", "Укажи URL страницы")
            return
        tok = os.environ.get("FIGMA_ACCESS_TOKEN") or os.environ.get("FIGMA_TOKEN")
        if not tok:
            messagebox.showerror("", "Нет FIGMA_ACCESS_TOKEN в окружении. Закрой приложение, в PowerShell:\n$env:FIGMA_ACCESS_TOKEN='...'\npython app.py")
            return
        try:
            thr = float(self.thr.get().replace(",", "."))
            ww = int(self.win_w.get().strip())
            wh = int(self.win_h.get().strip())
            sh = int(self.shift.get())
            sp = int(self.speck.get())
            px = int(self.pix_thr.get())
        except ValueError:
            messagebox.showwarning("", "Проверь числа")
            return

        c = load_cfg(self.cfg_path)
        fg = c.get("figma") or {}
        fk = (fg.get("file_key") or "").strip()
        nid = (fg.get("node_id") or "").strip()
        if not fk or not nid:
            messagebox.showerror("", "В config.json заполни секцию figma: file_key и node_id")
            return

        out_png = os.path.join(ROOT, fg.get("design_png", "storage/designs/figma_baseline_last.png"))
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

        fcfg = FigmaVsSiteConfig(
            site_url=site,
            figma_file_key=fk,
            figma_node_id=nid,
            figma_token=tok,
            figma_baseline_png=out_png,
            figma_scale=int(fg.get("scale", 1)),
            screenshot_dir=os.path.join(ROOT, c.get("screenshot_dir", "shots")),
            reports_dir=os.path.join(ROOT, c.get("reports_dir", "reports")),
            diff_threshold_pct=thr,
            ollama_url=c.get("ollama_url", "http://127.0.0.1:11434"),
            gemma_model=c.get("gemma_model", "gemma3"),
            use_gemma=self.use_gemma.get(),
            model_path=os.path.join(ROOT, c.get("model_path", "weights/diff_cnn.pt")),
            use_model=self.use_model.get(),
            window_size=(ww, wh),
            gemma_use_image=self.gemma_img.get(),
            tolerance_shift_px=max(0, min(5, sh)),
            tolerance_speckle_iter=max(0, min(5, sp)),
            pixel_threshold=max(0, min(255, px)),
        )

        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        self._append("— запуск —")
        self._busy = True
        self.btn.configure(state=tk.DISABLED)

        def job():
            def L(msg: str):
                self._q.put(msg)

            try:
                out = run_figma_vs_site(fcfg, log=L)
                self._q.put(("ok", out))
            except Exception as e:
                self._q.put(("err", e))

        threading.Thread(target=job, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
