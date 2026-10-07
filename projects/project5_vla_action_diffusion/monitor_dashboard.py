"""Read-only live dashboard for Project 5 AutoDL experiments.

The dashboard reads status.json, metrics.jsonl and eval_*.json files.  It has
no control endpoint, so it is safe to expose through a private SSH tunnel.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project 5 · AutoDL 实时监控</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2e;--line:#283450;--text:#eaf0ff;--muted:#8e9bb8;--blue:#55a7ff;--green:#45d19a;--amber:#f4bd65;--red:#ff7183}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#17254b 0,#0b1020 43%);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1240px;margin:auto;padding:28px 24px 44px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:22px}h1{font-size:25px;margin:0 0 5px}.sub,.small,th{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--green);white-space:nowrap}.warn{color:var(--amber)}.bad{color:var(--red)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px}.card{background:linear-gradient(145deg,#18213a,#11182a);border:1px solid var(--line);border-radius:13px;padding:16px;box-shadow:0 10px 28px #0002}.wide{grid-column:span 2}.full{grid-column:1/-1}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.7px}.value{font-size:25px;font-weight:700;margin-top:6px}.small{font-size:12px;margin-top:3px;word-break:break-word}.progress{height:10px;background:#0b1121;border-radius:20px;overflow:hidden;border:1px solid var(--line);margin:14px 0 8px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .5s}.row,.metric{display:flex;justify-content:space-between;gap:10px}.metric{align-items:center;border-bottom:1px solid #26314a;padding:7px 0}.metric:last-child{border-bottom:0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px 7px;border-bottom:1px solid #26314a}td.num{text-align:right}pre{max-height:220px;overflow:auto;white-space:pre-wrap;color:#b9c6e4;background:#0b1121;border-radius:9px;border:1px solid var(--line);padding:10px;font:12px/1.4 ui-monospace,Consolas,monospace;margin:0}.foot{margin-top:18px;color:var(--muted);font-size:12px;text-align:right}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.wide{grid-column:span 2}}@media(max-width:560px){main{padding:18px 12px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}header{display:block}.pill{display:inline-block;margin-top:12px}}
</style></head><body><main><header><div><h1>Project 5 · AutoDL 实时监控</h1><div class="sub" id="subtitle">正在连接…</div></div><div id="status" class="pill warn">CONNECTING</div></header>
<section class="grid"><div class="card"><div class="label">当前实验</div><div id="stage" class="value">—</div><div id="updated" class="small">—</div></div><div class="card"><div class="label">总进度</div><div id="progress" class="value">0 / 0</div><div id="progressText" class="small">等待数据</div></div><div class="card"><div class="label">训练 step</div><div id="step" class="value">—</div><div id="speed" class="small">—</div></div><div class="card"><div class="label">GPU</div><div id="gpu" class="value">—</div><div id="gpuSmall" class="small">—</div></div><div class="card wide"><h2>实验进度</h2><div class="row"><span id="detail">—</span><b id="percent">0%</b></div><div class="progress"><div id="bar" class="bar"></div></div><div class="small">页面每 2 秒刷新；服务只读，不提供训练控制。</div></div><div class="card wide"><h2>最近评估</h2><div id="results">尚无结果</div></div><div class="card wide"><h2>loss 尾部</h2><pre id="metrics">—</pre></div><div class="card wide"><h2>状态 JSON</h2><pre id="raw">—</pre></div></section><div class="foot">Project 5 · private SSH dashboard</div></main>
<script>const $=id=>document.getElementById(id);function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}function f(n,d=1){return n==null?'—':Number(n).toFixed(d)}function render(d){let s=d.status||{},g=d.gpu||{},p=d.progress||{};$('subtitle').textContent=(d.host||'AutoDL')+' · '+(d.root||'');$('status').textContent=(s.state||'unknown').toUpperCase();$('status').className='pill '+(s.state==='running'?'':s.state==='failed'?'bad':'warn');$('stage').textContent=s.run||'—';$('updated').textContent='更新于 '+new Date().toLocaleTimeString();$('progress').textContent=(p.done||0)+' / '+(p.total||0);$('percent').textContent=(p.percent||0)+'%';$('bar').style.width=(p.percent||0)+'%';$('progressText').textContent=p.detail||'等待数据';$('detail').textContent=s.method||'—';$('step').textContent=s.step==null?'—':s.step+' / '+(s.max_steps||'—');$('speed').textContent=s.avg100==null?'—':'loss(avg100) '+f(s.avg100,4);$('gpu').textContent=g.utilization==null?'—':f(g.utilization,0)+'%';$('gpuSmall').textContent=g.memory_used==null?'—':f(g.memory_used,0)+' / '+f(g.memory_total,0)+' MiB · '+f(g.temperature,0)+'°C';let rows=(d.results||[]).map(r=>'<tr><td>'+esc(r.name)+'</td><td class="num">'+f(r.success_rate*100,1)+'%</td><td class="num">'+f(r.collision_rate*100,1)+'%</td><td class="num">'+f(r.avg_steps,1)+'</td></tr>').join('');$('results').innerHTML=rows?'<table><thead><tr><th>实验</th><th>成功率</th><th>碰撞率</th><th>步数</th></tr></thead><tbody>'+rows+'</tbody></table>':'尚无结果';$('metrics').textContent=(d.metrics||[]).map(x=>JSON.stringify(x)).join('\n')||'—';$('raw').textContent=JSON.stringify(d,null,2)}async function refresh(){try{let r=await fetch('/api/status?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json())}catch(e){$('status').textContent='OFFLINE';$('status').className='pill bad';$('subtitle').textContent='无法读取监控服务：'+e}}refresh();setInterval(refresh,2000);</script></body></html>'''


def _gpu_info():
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
    try:
        values = [x.strip() for x in subprocess.check_output(cmd, text=True, timeout=2).splitlines()[0].split(",")]
        keys = ("utilization", "memory_used", "memory_total", "temperature")
        return {k: float(v) for k, v in zip(keys, values)}
    except (OSError, subprocess.SubprocessError, IndexError, ValueError):
        return {}


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _metrics(path, limit=12):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines]
    except (OSError, json.JSONDecodeError):
        return []


class Monitor:
    def __init__(self, root, total_runs=0):
        self.root = Path(root).resolve()
        self.total_runs = total_runs

    def snapshot(self):
        # Fair reruns are grouped as <condition>/seed_<n>. Discover actual run
        # directories by their status file while retaining compatibility with
        # the original one-level layout.
        runs = ([path.parent for path in self.root.rglob("status.json")
                 if "_queue" not in path.parts]
                if self.root.exists() else [])
        statuses = [(p, _read_json(p / "status.json")) for p in runs]
        active = [(p, s) for p, s in statuses if s.get("state") == "running"]
        current_path, current = (active[-1] if active else max(statuses, key=lambda x: x[0].stat().st_mtime, default=(None, {})))
        queue_path = self.root / "_queue" / "status.json"
        queue_status = _read_json(queue_path)
        if not active and queue_status.get("state") in {"waiting_gpu", "starting"}:
            current_path, current = queue_path.parent, queue_status
        results = []
        for p in runs:
            for path in p.glob("eval*.json"):
                payload = _read_json(path)
                if payload:
                    name = p.relative_to(self.root).as_posix() + "/" + path.stem
                    results.append({"name": name, **payload})
        done = sum(1 for _, s in statuses if s.get("state") == "completed")
        total = self.total_runs or max(len(runs), 1)
        detail = f"{len(runs)} 个运行目录，{done} 个已完成"
        if queue_status.get("state") == "waiting_gpu":
            detail = queue_status.get("message", detail)
        progress = {"done": done, "total": total, "percent": round(100 * done / total),
                    "detail": detail}
        return {"host": os.environ.get("HOSTNAME", os.environ.get("COMPUTERNAME", "AutoDL")),
                "root": str(self.root),
                "status": {"run": current_path.relative_to(self.root).as_posix()
                           if current_path else "—", **current},
                "progress": progress, "gpu": _gpu_info(),
                "metrics": _metrics(current_path / "metrics.jsonl") if current_path else [],
                "results": results}


def make_handler(monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            route = urlparse(self.path).path
            if route in {"/", "/index.html"}:
                body = HTML.encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            elif route == "/api/status":
                body = json.dumps(monitor.snapshot(), ensure_ascii=False).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
            else:
                self.send_error(404); return
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *_args):
            return
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="runs/project5")
    parser.add_argument("--total-runs", type=int, default=7)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(Monitor(args.run_root, args.total_runs)))
    print(f"Project 5 dashboard: http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
