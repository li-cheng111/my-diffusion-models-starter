"""Read-only live dashboard for the strict 200-epoch FID experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


STEP_RE = re.compile(r"step[_-](\d+)")
LOG_STEP_RE = re.compile(r"\[epoch\s+\d+\s+step\s+(\d+)\]\s+loss=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")
START_RE = re.compile(r"===== START (.+?) =====")
END_RE = re.compile(r"===== END (.+?) =====")

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DDPM FID 实验监控</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f4f6fa,#15171b);--card:light-dark(#fff,#20242c);--text:light-dark(#202633,#eef2f7);--muted:light-dark(#667085,#a7afbd);--line:light-dark(#dfe4ec,#363d49);--blue:light-dark(#356ae6,#8eafff);--amber:light-dark(#c87517,#f1b56e);--green:light-dark(#16834b,#72d6a2)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:25px 20px 42px}h1{margin:0;font-size:25px}.sub,.muted{color:var(--muted);font-size:13px}.sub{margin:7px 0 20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px}.label{color:var(--muted);font-size:12px}.value{font-size:20px;font-weight:650;margin-top:5px}.panel{margin-top:16px;overflow:auto}h2{font-size:16px;margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}.running,.sampling,.evaluating{color:var(--amber);font-weight:650}.completed{color:var(--green);font-weight:650}.pending,.stopped{color:var(--muted)}.bar{height:10px;min-width:130px;background:color-mix(in srgb,var(--blue) 15%,transparent);border-radius:99px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:inherit}.layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);gap:16px}.chart{width:100%;height:230px;display:block}.chart .axis,.chart .grid{stroke:var(--line);stroke-width:1}.chart .grid{stroke-dasharray:3 5}.chart .line{fill:none;stroke:var(--blue);stroke-width:2.5}.samples{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.samples img{width:100%;aspect-ratio:1;object-fit:contain;border:1px solid var(--line);background:var(--bg)}pre{max-height:250px;overflow:auto;white-space:pre-wrap;color:var(--muted);font-size:12px;line-height:1.45}.foot{margin-top:15px;color:var(--muted);font-size:12px}@media(max-width:760px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}.samples{grid-template-columns:repeat(4,1fr)}.value{font-size:17px}}
</style></head><body><main><h1>DDPM FID 改进实验实时监控</h1><div class="sub">严格 200 epoch · 每 2 秒刷新 · 页面只读，不干扰训练</div>
<div class="cards"><div class="card"><div class="label">总体状态</div><div id="status" class="value">连接中</div></div><div class="card"><div class="label">总体进度</div><div id="progress" class="value">—</div></div><div class="card"><div class="label">当前实验</div><div id="current" class="value">—</div></div><div class="card"><div class="label">GPU</div><div id="gpu" class="value">—</div></div></div>
<section class="panel"><h2>实验进度</h2><table><thead><tr><th>实验</th><th>状态</th><th>Step</th><th>进度</th><th>Loss</th><th>LR</th><th>EMA</th><th>Checkpoint</th></tr></thead><tbody id="runs"></tbody></table></section>
<div class="layout"><section class="panel"><h2 id="loss-title">Loss 曲线</h2><svg id="chart" class="chart" viewBox="0 0 900 230"><line class="axis" x1="55" y1="200" x2="875" y2="200"/><line class="axis" x1="55" y1="18" x2="55" y2="200"/><g id="grid"></g><polyline id="line" class="line" points=""/><text x="465" y="225" text-anchor="middle">training step</text></svg></section><section class="panel"><h2>最新样本</h2><div id="samples" class="samples"></div><div id="sample-empty" class="muted">尚未生成样本</div></section></div>
<section class="panel"><h2>当前日志</h2><pre id="log">—</pre></section><div id="foot" class="foot">—</div></main>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function draw(rows){if(!rows||rows.length<2){$('line').setAttribute('points','');return}let x0=55,x1=875,y0=18,y1=200,ss=rows.map(r=>+r.step),vv=rows.map(r=>+r.loss),a=Math.min(...ss),b=Math.max(...ss),lo=Math.min(...vv),hi=Math.max(...vv),d=Math.max(hi-lo,1e-8);let x=v=>x0+(v-a)/Math.max(b-a,1)*(x1-x0),y=v=>y1-(v-lo)/d*(y1-y0);$('line').setAttribute('points',rows.map(r=>`${x(+r.step).toFixed(1)},${y(+r.loss).toFixed(1)}`).join(' '));$('grid').innerHTML=`<text x="47" y="22" text-anchor="end" fill="var(--muted)" font-size="12">${hi.toFixed(3)}</text><text x="47" y="204" text-anchor="end" fill="var(--muted)" font-size="12">${lo.toFixed(3)}</text><text x="55" y="218" fill="var(--muted)" font-size="12">${a.toLocaleString()}</text><text x="875" y="218" text-anchor="end" fill="var(--muted)" font-size="12">${b.toLocaleString()}</text>`}
function render(d){$('status').textContent=d.status_label;$('current').textContent=d.current||'—';$('gpu').textContent=d.gpu.label;$('progress').textContent=`${d.completed_steps.toLocaleString()} / ${d.total_steps.toLocaleString()} (${d.percent.toFixed(1)}%)`;$('runs').innerHTML=d.runs.map(r=>`<tr><td>${esc(r.label)}</td><td class="${r.status}">${esc(r.status_label)}</td><td>${r.step.toLocaleString()} / ${r.total_steps.toLocaleString()}</td><td><div class="bar"><div class="fill" style="width:${r.percent}%"></div></div>${r.percent.toFixed(1)}%</td><td>${r.latest_loss??'—'}</td><td>${r.lr??'—'}</td><td>${esc(r.ema||'—')}</td><td>${esc(r.checkpoint||'—')}</td></tr>`).join('');let active=d.runs.find(r=>r.label===d.current)||d.runs[0];$('loss-title').textContent=`Loss 曲线 · ${active?.label||'—'}`;draw(active?.loss||[]);let samples=active?.samples||[];$('sample-empty').hidden=samples.length>0;$('samples').innerHTML=samples.map(s=>`<img src="${esc(s.url)}?v=${s.mtime}" alt="step ${s.step}">`).join('');$('log').textContent=d.log||'暂无日志';$('foot').textContent=`最后更新：${d.observed_at} · 自动刷新：2 秒`}
async function refresh(){try{let r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);render(await r.json())}catch(e){$('status').textContent='监控服务连接失败';$('log').textContent=e.message}}refresh();setInterval(refresh,2000);
</script></body></html>'''


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    total_steps: int


def _step(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _loss(path: Path, limit: int = 400) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))[-limit:]
        return [
            {"step": float(row["step"]), "loss": float(row["loss"]), "lr": float(row.get("lr", 0.0))}
            for row in rows
            if row.get("loss") is not None
        ]
    except (OSError, KeyError, TypeError, ValueError):
        return []


def _log_state(text: str, limit: int = 400) -> tuple[str | None, dict[str, list[dict[str, Any]]]]:
    """Extract the active run and live training points from the runner log."""

    def canonical_label(name: str) -> str:
        return name.strip().removesuffix("_RECOVERY")

    current: str | None = None
    points: dict[str, list[dict[str, Any]]] = {}
    for line in text.splitlines():
        start = START_RE.search(line)
        if start:
            current = canonical_label(start.group(1))
            points.setdefault(current, [])
            continue
        end = END_RE.search(line)
        if end:
            if current == canonical_label(end.group(1)):
                current = None
            continue
        if current is None:
            continue
        match = LOG_STEP_RE.search(line)
        if match:
            points.setdefault(current, []).append(
                {
                    "step": float(match.group(1)),
                    "loss": float(match.group(2)),
                    "lr": float(match.group(3)),
                }
            )
    return current, {label: rows[-limit:] for label, rows in points.items()}


def _gpu() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        usage, used, total = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")[:3]]
        return {"label": f"{usage}% · {used}/{total} MiB", "usage": usage, "used": used, "total": total}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {"label": "unavailable", "usage": "—", "used": "—", "total": "—"}


def _pid_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


class Dashboard:
    def __init__(self, specs: list[RunSpec], pid_file: Path, log_path: Path) -> None:
        self.specs = specs
        self.pid_file = pid_file
        self.log_path = log_path

    def snapshot(self) -> dict[str, Any]:
        try:
            log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        active_label, live_loss = _log_state(log_text)
        runner_alive = _pid_alive(self.pid_file)
        runs = []
        for spec in self.specs:
            ckpt = spec.path / "ckpt"
            loss_path = spec.path / "loss_history.csv"
            loss = _loss(loss_path)
            logged_loss = live_loss.get(spec.label, [])
            if logged_loss and (not loss or logged_loss[-1]["step"] >= loss[-1]["step"]):
                loss = logged_loss
            steps = [int(row["step"]) for row in loss]
            steps.extend(_step(path) for path in ckpt.glob("step_*.pt"))
            steps.extend(_step(path) for path in (spec.path / "samples").glob("step_*.png"))
            # The training code writes loss_history only after the final
            # checkpoint succeeds. Requiring both files prevents a truncated
            # final.pt left by a full disk from being reported as completed.
            final = (ckpt / "final.pt").exists() and loss_path.is_file()
            step = spec.total_steps if final else max(steps, default=0)
            active = runner_alive and active_label == spec.label
            failed = f"{spec.label} training failed" in log_text
            status = "evaluating" if final and active else (
                "completed" if final else (
                    "running" if active else ("failed" if failed else ("stopped" if step else "pending"))
                )
            )
            latest = loss[-1] if loss else {}
            sample_paths = sorted((spec.path / "samples").glob("step_*.png"), key=_step)[-8:]
            checkpoint_paths = list(ckpt.glob("*.pt"))
            latest_checkpoint = (
                "final.pt" if final else max(checkpoint_paths, key=_step).name if checkpoint_paths else None
            )
            runs.append({
                "label": spec.label, "status": status,
                "status_label": {"completed":"已完成","running":"训练中","evaluating":"评估中","failed":"失败","stopped":"已停止","pending":"等待中"}[status],
                "step": step, "total_steps": spec.total_steps,
                "percent": min(100.0, 100.0 * step / max(spec.total_steps, 1)),
                "latest_loss": f"{latest['loss']:.5f}" if latest else None,
                "lr": f"{latest['lr']:.3e}" if latest and latest.get("lr") else None,
                "ema": "bank: 0.999 / 0.9995 / 0.9999",
                "checkpoint": latest_checkpoint,
                "loss": loss,
                "samples": [{"step": _step(path), "url": f"/sample?run={spec.label}&name={path.name}", "mtime": path.stat().st_mtime_ns} for path in sample_paths],
            })
        completed = sum(run["status"] == "completed" for run in runs)
        failed = sum(run["status"] == "failed" for run in runs)
        current = next((run for run in runs if run["status"] in {"running", "evaluating"}), None)
        if current is None:
            current = next((run for run in runs if run["status"] == "pending"), None)
        if current is None:
            current = next((run for run in runs if run["status"] == "failed"), None)
        total_steps = sum(spec.total_steps for spec in self.specs)
        completed_steps = sum(int(run["step"]) for run in runs)
        log = "\n".join(log_text.splitlines()[-24:])
        return {
            "status": "completed" if completed == len(runs) and runs else (
                current["status"] if current and current["status"] in {"running", "evaluating"}
                else ("failed" if failed else "pending")
            ),
            "status_label": "全部完成" if completed == len(runs) and runs else (
                "训练中" if current and current["status"] == "running" else (
                    "评估中" if current and current["status"] == "evaluating"
                    else ("有实验失败" if failed else "等待启动")
                )
            ),
            "current": current["label"] if current else None,
            "completed_steps": completed_steps, "total_steps": total_steps,
            "percent": min(100.0, 100.0 * completed_steps / max(total_steps, 1)),
            "completed_runs": completed, "total_runs": len(runs), "runs": runs,
            "gpu": _gpu(), "log": log,
            "observed_at": __import__("datetime").datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

    def sample_path(self, run_label: str, name: str) -> Path | None:
        for spec in self.specs:
            if spec.label != run_label:
                continue
            candidate = (spec.path / "samples" / unquote(name)).resolve()
            if candidate.parent == (spec.path / "samples").resolve() and candidate.suffix == ".png" and candidate.is_file():
                return candidate
        return None


def make_handler(dashboard: Dashboard):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/progress":
                body, content_type = json.dumps(dashboard.snapshot(), ensure_ascii=False).encode(), "application/json; charset=utf-8"
            elif parsed.path == "/sample":
                params = parse_qs(parsed.query)
                path = dashboard.sample_path(params.get("run", [""])[0], params.get("name", [""])[0])
                if path is None:
                    self.send_error(404)
                    return
                body, content_type = path.read_bytes(), "image/png"
            else:
                body, content_type = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=path, repeatable")
    parser.add_argument("--total-steps", type=int, default=78000)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    specs = []
    for item in args.run:
        if "=" not in item:
            parser.error("--run must have the form label=path")
        label, path = item.split("=", 1)
        specs.append(RunSpec(label, Path(path).resolve(), args.total_steps))
    dashboard = Dashboard(specs, args.pid_file.resolve(), args.log_path.resolve())
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dashboard))
    print(f"FID monitor: http://{args.host}:{args.port}/", flush=True)
    print("Watching: " + ", ".join(f"{spec.label}={spec.path}" for spec in specs), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
