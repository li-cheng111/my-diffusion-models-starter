"""Read-only live dashboard for the strict 200-epoch FID experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


STEP_RE = re.compile(r"step[_-](\d+)")
LOG_STEP_RE = re.compile(r"\[epoch\s+\d+\s+step\s+(\d+)\]\s+loss=([0-9.eE+-]+)\s+lr=([0-9.eE+-]+)")
START_RE = re.compile(r"===== START (.+?) =====")
END_RE = re.compile(r"===== END (.+?) =====")
EMA_DECAYS = ("0.999", "0.9995", "0.9999")
FID_SAMPLE_COUNT = 5000

PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>DDPM FID 实验监控</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f4f6fa,#15171b);--card:light-dark(#fff,#20242c);--text:light-dark(#202633,#eef2f7);--muted:light-dark(#667085,#a7afbd);--line:light-dark(#dfe4ec,#363d49);--blue:light-dark(#356ae6,#8eafff);--amber:light-dark(#c87517,#f1b56e);--green:light-dark(#16834b,#72d6a2)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1240px;margin:auto;padding:25px 20px 42px}h1{margin:0;font-size:25px}.sub,.muted{color:var(--muted);font-size:13px}.sub{margin:7px 0 20px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px}.label{color:var(--muted);font-size:12px}.value{font-size:20px;font-weight:650;margin-top:5px}.panel{margin-top:16px;overflow:auto}h2{font-size:16px;margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}.running,.sampling,.evaluating,.preparing{color:var(--amber);font-weight:650}.completed{color:var(--green);font-weight:650}.pending,.stopped{color:var(--muted)}.bar{height:10px;min-width:130px;background:color-mix(in srgb,var(--blue) 15%,transparent);border-radius:99px;overflow:hidden}.fill{height:100%;background:var(--blue);border-radius:inherit}.layout{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(280px,.8fr);gap:16px}.chart{width:100%;height:230px;display:block}.chart .axis,.chart .grid{stroke:var(--line);stroke-width:1}.chart .grid{stroke-dasharray:3 5}.chart .line{fill:none;stroke:var(--blue);stroke-width:2.5}.samples{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.samples img{width:100%;aspect-ratio:1;object-fit:contain;border:1px solid var(--line);background:var(--bg)}pre{max-height:250px;overflow:auto;white-space:pre-wrap;color:var(--muted);font-size:12px;line-height:1.45}.foot{margin-top:15px;color:var(--muted);font-size:12px}@media(max-width:760px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}.samples{grid-template-columns:repeat(4,1fr)}.value{font-size:17px}}
</style><style>.eval-list{display:grid;gap:7px;min-width:255px}.eval-row{display:grid;gap:3px}.eval-meta{display:flex;justify-content:space-between;gap:10px;font-size:12px}.eval-label{color:var(--muted)}.eval-bar{height:6px;min-width:0}.eval-bar .evaluating{background:var(--amber)}.eval-bar .completed{background:var(--green)}.eval-bar .failed{background:#d14343}.eval-bar .stopped{background:var(--muted)}.ready{color:var(--muted)}.failed{color:#d14343;font-weight:650}</style></head><body><main><h1>DDPM FID 改进实验实时监控</h1><div class="sub">线性 / cosine 对照 · 每 2 秒刷新 · 页面只读，不干扰训练</div>
<div class="cards"><div class="card"><div class="label">总体状态</div><div id="status" class="value">连接中</div></div><div class="card"><div class="label">总体进度</div><div id="progress" class="value">—</div></div><div class="card"><div class="label">当前实验</div><div id="current" class="value">—</div></div><div class="card"><div class="label">GPU</div><div id="gpu" class="value">—</div></div></div>
<section class="panel"><h2>实验进度</h2><table><thead><tr><th>实验</th><th>状态</th><th>Step</th><th>进度</th><th>Loss</th><th>LR</th><th>FID / EMA 评估</th><th>Checkpoint</th></tr></thead><tbody id="runs"></tbody></table></section>
<div class="layout"><section class="panel"><h2 id="loss-title">Loss 曲线</h2><svg id="chart" class="chart" viewBox="0 0 900 230"><line class="axis" x1="55" y1="200" x2="875" y2="200"/><line class="axis" x1="55" y1="18" x2="55" y2="200"/><g id="grid"></g><polyline id="line" class="line" points=""/><text x="465" y="225" text-anchor="middle">training step</text></svg></section><section class="panel"><h2>最新样本</h2><div id="samples" class="samples"></div><div id="sample-empty" class="muted">尚未生成样本</div></section></div>
<section class="panel"><h2>当前日志</h2><pre id="log">—</pre></section><div id="foot" class="foot">—</div></main>
<script>
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function draw(rows){if(!rows||rows.length<2){$('line').setAttribute('points','');return}let x0=55,x1=875,y0=18,y1=200,ss=rows.map(r=>+r.step),vv=rows.map(r=>+r.loss),a=Math.min(...ss),b=Math.max(...ss),lo=Math.min(...vv),hi=Math.max(...vv),d=Math.max(hi-lo,1e-8);let x=v=>x0+(v-a)/Math.max(b-a,1)*(x1-x0),y=v=>y1-(v-lo)/d*(y1-y0);$('line').setAttribute('points',rows.map(r=>`${x(+r.step).toFixed(1)},${y(+r.loss).toFixed(1)}`).join(' '));$('grid').innerHTML=`<text x="47" y="22" text-anchor="end" fill="var(--muted)" font-size="12">${hi.toFixed(3)}</text><text x="47" y="204" text-anchor="end" fill="var(--muted)" font-size="12">${lo.toFixed(3)}</text><text x="55" y="218" fill="var(--muted)" font-size="12">${a.toLocaleString()}</text><text x="875" y="218" text-anchor="end" fill="var(--muted)" font-size="12">${b.toLocaleString()}</text>`}
function render(d){$('status').textContent=d.status_label;$('status').className=`value ${d.status}`;$('current').textContent=d.current_label||d.current||'—';$('gpu').textContent=d.gpu.label;$('progress').textContent=d.dataset?.downloading?`数据下载中 · ${d.dataset.downloaded_mb.toFixed(1)} MB`:`${d.completed_steps.toLocaleString()} / ${d.total_steps.toLocaleString()} steps · FID ${d.completed_evaluations||0}/${d.total_evaluations||0}`;$('runs').innerHTML=d.runs.map(r=>{const evals=(r.evaluations||[]).map(e=>`<div class="eval-row"><div class="eval-meta"><span>EMA ${esc(e.decay)}</span><span class="${esc(e.status)}">${esc(e.status_label)}</span></div><div class="bar eval-bar"><div class="fill ${esc(e.status)}" style="width:${Math.max(0,Math.min(100,e.percent||0))}%"></div></div></div>`).join('');return `<tr><td>${esc(r.label)}</td><td class="${r.status}">${esc(r.status_label)}</td><td>${r.step.toLocaleString()} / ${r.total_steps.toLocaleString()}</td><td><div class="bar"><div class="fill" style="width:${r.percent}%"></div></div>${r.percent.toFixed(1)}%</td><td>${r.latest_loss??'—'}</td><td>${r.lr??'—'}</td><td><div class="eval-list">${evals||'<span class="eval-label">暂无评估记录</span>'}</div></td><td>${esc(r.checkpoint||'—')}</td></tr>`}).join('');let active=d.runs.find(r=>r.label===d.current)||d.runs[0];$('loss-title').textContent=`Loss 曲线 · ${active?.label||'—'}`;draw(active?.loss||[]);let samples=active?.samples||[];$('sample-empty').hidden=samples.length>0;$('samples').innerHTML=samples.map(s=>`<img src="${esc(s.url)}?v=${s.mtime}" alt="step ${s.step}">`).join('');$('log').textContent=d.log||'暂无日志';$('foot').textContent=`最后更新：${d.observed_at} · 自动刷新：2 秒`}
async function refresh(){try{let r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);render(await r.json())}catch(e){$('status').textContent='监控服务连接失败';$('log').textContent=e.message}}refresh();setInterval(refresh,2000);
</script></body></html>'''


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    total_steps: int


def _step(path: Path) -> int:
    if path.stem == "grid":
        return 10**12
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


_GPU_CACHE_TTL_SECONDS = 30.0
_GPU_CACHE_LOCK = threading.Lock()
_GPU_CACHE: tuple[float, dict[str, str]] | None = None


def _probe_gpu() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        usage, used, total = [part.strip() for part in result.stdout.strip().splitlines()[0].split(",")[:3]]
        return {"label": f"{usage}% · {used}/{total} MiB", "usage": usage, "used": used, "total": total}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        pass

    for command in ("musa-smi", "mthreads-smi", "/usr/local/musa/bin/musaInfo"):
        try:
            result = subprocess.run(
                [command], capture_output=True, text=True, timeout=2, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        total_match = re.search(r"memInfo\.total:\s*([\d.]+)\s*GB", result.stdout)
        free_match = re.search(r"memInfo\.free:\s*([\d.]+)\s*GB", result.stdout)
        if total_match and free_match:
            total = float(total_match.group(1))
            used = max(0.0, total - float(free_match.group(1)))
            return {
                "label": f"MUSA · 利用率不可读 · {used:.2f}/{total:.2f} GB",
                "usage": "—",
                "used": f"{used:.2f} GB",
                "total": f"{total:.2f} GB",
            }
    return {"label": "unavailable", "usage": "—", "used": "—", "total": "—"}


def _gpu() -> dict[str, str]:
    """Cache hardware telemetry so multiple dashboard tabs don't spawn probes every 2s."""
    global _GPU_CACHE

    now = time.monotonic()
    cached = _GPU_CACHE
    if cached is not None and now - cached[0] < _GPU_CACHE_TTL_SECONDS:
        return dict(cached[1])

    # The MUSA fallback (musaInfo) can be slow. Serialize cache misses so
    # concurrent requests from several open monitor tabs share one probe.
    with _GPU_CACHE_LOCK:
        now = time.monotonic()
        cached = _GPU_CACHE
        if cached is not None and now - cached[0] < _GPU_CACHE_TTL_SECONDS:
            return dict(cached[1])

        value = _probe_gpu()
        _GPU_CACHE = (time.monotonic(), value)
        return dict(value)


def _pid_alive(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _active_evaluations(specs: list[RunSpec]) -> dict[str, str]:
    """Return active EMA evaluations by run label, based on live Python processes."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "comm=,args="],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    active: dict[str, str] = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or Path(fields[0]).name not in {"python", "python3"}:
            continue
        args = fields[1]
        if "evaluate.py" not in args:
            continue
        decay_match = re.search(r"--ema_decay\s+([0-9.]+)", args)
        if not decay_match:
            continue
        for spec in specs:
            if str(spec.path.resolve()) in args:
                active[spec.label] = decay_match.group(1)
                break
    return active


def _evaluation_states(run_path: Path, active_decay: str | None) -> list[dict[str, Any]]:
    """Read per-EMA FID logs and result files without touching the evaluator."""
    ckpt_dir = run_path / "ckpt"
    states = []
    for decay in EMA_DECAYS:
        suffix = decay.replace(".", "")
        log_path = run_path / "evaluations" / f"fid_ema_{suffix}.log"
        metric_name = f"EMA_{float(decay):.4f}"
        metric_path = ckpt_dir / f"fid_{FID_SAMPLE_COUNT}_{metric_name}_clipx0.txt"
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        except OSError:
            log_text = ""
        try:
            metric_text = metric_path.read_text(encoding="utf-8", errors="replace") if metric_path.is_file() else ""
        except OSError:
            metric_text = ""

        score_match = re.search(r"FID:\s*([0-9.]+)", metric_text)
        if score_match is None:
            score_match = re.search(r"FID\s*@\s*\d+\s*samples.*?:\s*([0-9.]+)", log_text)
        real_matches = re.findall(r"FID real features:\s*(\d+)/(\d+)", log_text)
        generated_matches = re.findall(r"FID generated samples:\s*(\d+)/(\d+)", log_text)
        real_done, real_total = tuple(map(int, real_matches[-1])) if real_matches else (0, FID_SAMPLE_COUNT)
        gen_done, gen_total = tuple(map(int, generated_matches[-1])) if generated_matches else (0, FID_SAMPLE_COUNT)

        if score_match:
            status, percent = "completed", 100.0
            status_label = f"FID {float(score_match.group(1)):.4f}"
        elif active_decay == decay:
            status = "evaluating"
            if gen_done >= gen_total:
                percent, status_label = 100.0, "计算 FID"
            elif generated_matches:
                percent = 100.0 * gen_done / max(gen_total, 1)
                status_label = f"生成 {gen_done:,}/{gen_total:,}"
            elif real_matches:
                percent = 100.0 * real_done / max(real_total, 1)
                status_label = f"真实特征 {real_done:,}/{real_total:,}"
            else:
                percent, status_label = 0.0, "加载模型/数据"
        elif log_text and any(marker in log_text for marker in ("Traceback", "evaluation failed:", "RuntimeError:")):
            status, percent, status_label = "failed", 0.0, "评估失败"
        elif log_text:
            status, percent, status_label = "stopped", 0.0, "进程未运行"
        else:
            status, percent, status_label = "pending", 0.0, "等待评估"

        states.append({
            "decay": decay, "status": status, "status_label": status_label,
            "percent": round(max(0.0, min(100.0, percent)), 1),
            "real_done": real_done, "real_total": real_total,
            "generated_done": gen_done, "generated_total": gen_total,
            "score": float(score_match.group(1)) if score_match else None,
            "log_tail": "\n".join(log_text.splitlines()[-18:]),
        })
    return states


class Dashboard:
    def __init__(
        self,
        specs: list[RunSpec],
        pid_file: Path,
        log_path: Path,
        dataset_dir: Path | None = None,
    ) -> None:
        self.specs = specs
        self.pid_file = pid_file
        self.log_path = log_path
        self.dataset_dir = dataset_dir

    def snapshot(self) -> dict[str, Any]:
        try:
            log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        active_label, live_loss = _log_state(log_text)
        runner_alive = _pid_alive(self.pid_file)
        active_evaluations = _active_evaluations(self.specs)
        archive = self.dataset_dir / "cifar-10-python.tar.gz" if self.dataset_dir else None
        dataset_ready = bool(self.dataset_dir and (self.dataset_dir / "cifar-10-batches-py").is_dir())
        dataset_bytes = 0
        if archive and archive.is_file() and not dataset_ready:
            try:
                dataset_bytes = archive.stat().st_size
            except OSError:
                pass
        preparing_dataset = (
            runner_alive
            and active_label is None
            and self.dataset_dir is not None
            and not dataset_ready
            and (dataset_bytes > 0 or "CIFAR-10 will be downloaded" in log_text)
        )
        runs = []
        for spec in self.specs:
            ckpt = spec.path / "ckpt"
            evaluations = _evaluation_states(spec.path, active_evaluations.get(spec.label))
            active_evaluation = next((item for item in evaluations if item["status"] == "evaluating"), None)
            loss_path = spec.path / "loss_history.csv"
            loss = _loss(loss_path)
            logged_loss = live_loss.get(spec.label, [])
            # A recovery run can append a fresh attempt after an older
            # loss_history.csv stopped at a larger step.  While that attempt
            # is active, its runner log is the authoritative live source.
            if logged_loss and (active_label == spec.label or not loss or logged_loss[-1]["step"] >= loss[-1]["step"]):
                loss = logged_loss
            steps = [int(row["step"]) for row in loss]
            steps.extend(_step(path) for path in ckpt.glob("step_*.pt"))
            steps.extend(_step(path) for path in (spec.path / "samples").glob("step_*.png"))
            # The training code writes loss_history only after the final
            # checkpoint succeeds. Requiring both files prevents a truncated
            # final.pt left by a full disk from being reported as completed.
            final_checkpoint = next(
                (path for path in (ckpt / "final.pt", ckpt / "epoch_latest.pt") if path.is_file()),
                None,
            )
            final = final_checkpoint is not None and loss_path.is_file()
            step = spec.total_steps if final else max(steps, default=0)
            active = runner_alive and active_label == spec.label
            train_failed = any(
                line.startswith(f"{spec.label} training failed:")
                or (
                    line.startswith(f"{spec.label} EMA ")
                    and " evaluation failed:" in line
                )
                or line.startswith(f"{spec.label} sample generation failed:")
                for line in log_text.splitlines()
            )
            eval_failed = any(item["status"] == "failed" for item in evaluations)
            eval_completed = all(item["status"] == "completed" for item in evaluations)
            eval_stopped = any(item["status"] == "stopped" for item in evaluations)
            failed = train_failed or eval_failed
            if active_evaluation:
                status = "evaluating"
            elif active:
                status = "running"
            elif failed:
                status = "failed"
            elif final and eval_completed:
                status = "completed"
            elif final and eval_stopped:
                status = "stopped"
            elif final:
                status = "ready"
            else:
                status = "stopped" if step else "pending"
            latest = loss[-1] if loss else {}
            sample_paths = sorted(
                [* (spec.path / "samples").glob("step_*.png"), spec.path / "samples" / "grid.png"],
                key=_step,
            )[-8:]
            sample_paths = [path for path in sample_paths if path.is_file()]
            checkpoint_paths = list(ckpt.glob("*.pt"))
            latest_checkpoint = (
                final_checkpoint.name if final else max(checkpoint_paths, key=_step).name if checkpoint_paths else None
            )
            runs.append({
                "label": spec.label, "status": status,
                "status_label": {"completed":"训练与 FID 完成","running":"训练中","evaluating":"FID 评估中","failed":"失败","stopped":"评估中断","ready":"训练完成·待评估","pending":"等待中"}[status],
                "step": step, "total_steps": spec.total_steps,
                "percent": min(100.0, 100.0 * step / max(spec.total_steps, 1)),
                "latest_loss": f"{latest['loss']:.5f}" if latest else None,
                "lr": f"{latest['lr']:.3e}" if latest and latest.get("lr") else None,
                "ema": "bank: 0.999 / 0.9995 / 0.9999",
                "checkpoint": latest_checkpoint,
                "evaluations": evaluations,
                "active_ema": active_evaluation["decay"] if active_evaluation else None,
                "loss": loss,
                "samples": [{"step": _step(path), "url": f"/sample?run={spec.label}&name={path.name}", "mtime": path.stat().st_mtime_ns} for path in sample_paths],
            })
        completed = sum(run["status"] == "completed" for run in runs)
        completed_evaluations = sum(
            item["status"] == "completed" for run in runs for item in run["evaluations"]
        )
        total_evaluations = sum(len(run["evaluations"]) for run in runs)
        failed = sum(run["status"] == "failed" for run in runs)
        current = next((run for run in runs if run["status"] in {"running", "evaluating"}), None)
        if current is None:
            current = next((run for run in runs if run["status"] == "pending"), None)
        if current is None:
            current = next((run for run in runs if run["status"] == "failed"), None)
        if current is None:
            current = next((run for run in runs if run["status"] == "ready"), None)
        total_steps = sum(spec.total_steps for spec in self.specs)
        completed_steps = sum(int(run["step"]) for run in runs)
        log = "\n".join(log_text.splitlines()[-24:])
        overall_status = "completed" if completed == len(runs) and runs else (
            current["status"] if current else ("failed" if failed else "pending")
        )
        overall_label = "全部训练与 FID 评估完成" if completed == len(runs) and runs else (
            "训练中" if current and current["status"] == "running" else (
                "FID 评估中" if current and current["status"] == "evaluating" else (
                    "训练完成·等待评估" if current and current["status"] == "ready" else (
                        "评估中断" if current and current["status"] == "stopped" else (
                            "有实验失败" if failed else "等待启动"
                        )
                    )
                )
            )
        )
        current_label = current["label"] if current else None
        if current and current.get("active_ema"):
            current_label = f"{current_label} · EMA {current['active_ema']}"
            active_eval_state = next(
                item for item in current["evaluations"] if item["decay"] == current["active_ema"]
            )
            log = active_eval_state.get("log_tail") or log
        dataset = {"downloading": False, "downloaded_mb": 0.0}
        if preparing_dataset:
            overall_status = "preparing"
            overall_label = "下载 CIFAR-10"
            current_label = "准备训练数据"
            dataset = {"downloading": True, "downloaded_mb": dataset_bytes / (1024 * 1024)}
            log = f"CIFAR-10 下载中：{dataset['downloaded_mb']:.1f} MB\n" + log
        return {
            "status": overall_status,
            "status_label": overall_label,
            "current": current["label"] if current else None,
            "current_label": current_label,
            "completed_steps": completed_steps, "total_steps": total_steps,
            "percent": min(100.0, 100.0 * completed_steps / max(total_steps, 1)),
            "completed_runs": completed, "total_runs": len(runs), "runs": runs,
            "completed_evaluations": completed_evaluations,
            "total_evaluations": total_evaluations,
            "gpu": _gpu(), "log": log, "dataset": dataset,
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
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    specs = []
    for item in args.run:
        if "=" not in item:
            parser.error("--run must have the form label=path")
        label, path = item.split("=", 1)
        specs.append(RunSpec(label, Path(path).resolve(), args.total_steps))
    dashboard = Dashboard(
        specs,
        args.pid_file.resolve(),
        args.log_path.resolve(),
        args.dataset_dir.resolve() if args.dataset_dir else None,
    )
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
