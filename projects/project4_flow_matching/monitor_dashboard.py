"""Read-only live training dashboard for AutoDL; serves on port 6006 by default."""
import argparse
import csv
import json
import os
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Project 4 实时训练监控</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2e;--line:#293650;--text:#eaf0ff;--muted:#91a0bf;--blue:#64b5ff;--green:#45d19a;--amber:#f4bd65}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#17254b 0,#0b1020 44%);color:var(--text);font:14px/1.5 system-ui,Segoe UI,sans-serif}main{max-width:1180px;margin:auto;padding:28px 20px}header,.split,.row{display:flex;justify-content:space-between;gap:16px;align-items:center}h1{font-size:25px;margin:0 0 5px}h2{font-size:16px;margin:0 0 12px}.muted,.small{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:20px;padding:6px 12px;color:var(--green)}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0}.card{background:linear-gradient(145deg,#18213a,#11182a);border:1px solid var(--line);border-radius:13px;padding:16px;min-width:0}.wide{grid-column:span 2}.full{grid-column:1/-1}.label{font-size:12px;color:var(--muted)}.value{font-size:24px;font-weight:700;margin:5px 0}.progress{height:10px;border:1px solid var(--line);background:#0a1020;border-radius:10px;overflow:hidden;margin:12px 0 7px}.bar{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .5s}.metric{display:flex;justify-content:space-between;border-bottom:1px solid #28334b;padding:6px 0}.metric:last-child{border:0}canvas{width:100%;height:220px}pre{max-height:240px;overflow:auto;white-space:pre-wrap;word-break:break-all;background:#0a1020;border:1px solid var(--line);border-radius:8px;padding:12px;color:#c2cee7;font:12px/1.45 ui-monospace,Consolas,monospace}.runs{display:flex;gap:9px;flex-wrap:wrap}.run{border:1px solid var(--line);border-radius:20px;padding:5px 10px}.foot{text-align:right;color:var(--muted);font-size:12px;margin:16px 0}@media(max-width:760px){.grid{grid-template-columns:repeat(2,1fr)}.wide{grid-column:span 2}}@media(max-width:500px){main{padding:18px 12px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}header{align-items:flex-start;flex-direction:column}}
</style></head><body><main>
<header><div><h1>Project 4 · Flow Matching 实时监控</h1><div class="muted" id="host">连接中…</div></div><div class="pill" id="state">CONNECTING</div></header>
<div class="runs" id="runs"></div><section class="grid">
<div class="card"><div class="label">当前阶段</div><div class="value" id="stage">—</div><div class="small" id="updated">—</div></div>
<div class="card"><div class="label">训练步数</div><div class="value" id="steps">—</div><div class="small" id="pct">—</div><div class="progress"><div class="bar" id="bar"></div></div></div>
<div class="card"><div class="label">速度 / 预计剩余</div><div class="value" id="speed">—</div><div class="small" id="eta">—</div></div>
<div class="card"><div class="label">Loss / 学习率</div><div class="value" id="loss">—</div><div class="small" id="lr">—</div></div>
<div class="card wide"><h2>训练 Loss</h2><canvas id="chart"></canvas></div>
<div class="card wide"><h2>GPU / 检查点</h2><div id="gpu"></div></div>
<div class="card full"><h2>最近日志</h2><pre id="log">等待训练日志…</pre></div>
</section><div class="foot">只读页面 · 每 2 秒刷新 · 更新时间 <span id="clock">—</span></div></main>
<script>
const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function duration(s){if(!Number.isFinite(+s))return '—';s=Math.max(0,+s);let h=Math.floor(s/3600),m=Math.floor(s%3600/60);return h?`${h}小时${m}分`:`${m}分钟`}
function chart(points){let c=$('chart'),ctx=c.getContext('2d'),w=c.clientWidth,h=220,dpr=devicePixelRatio||1;c.width=w*dpr;c.height=h*dpr;ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);if(points.length<2){ctx.fillStyle='#91a0bf';ctx.fillText('累计 2 个日志点后显示曲线',12,24);return}let vals=points.map(p=>p.loss),lo=Math.min(...vals),hi=Math.max(...vals),pad=20;ctx.strokeStyle='#293650';ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,h-pad);ctx.lineTo(w-pad,h-pad);ctx.stroke();ctx.strokeStyle='#64b5ff';ctx.lineWidth=2;ctx.beginPath();points.forEach((p,i)=>{let x=pad+(w-2*pad)*i/(points.length-1),y=pad+(h-2*pad)*(1-(p.loss-lo)/(hi-lo||1));i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.fillStyle='#91a0bf';ctx.fillText(`loss ${lo.toFixed(3)} – ${hi.toFixed(3)}`,pad+4,pad+12)}
function render(d){let t=d.training||{},g=d.gpu||{};$('host').textContent=`${d.host} · ${d.run_dir}`;$('state').textContent=(d.state||'waiting').toUpperCase();$('stage').textContent=d.stage||'等待启动';$('updated').textContent=`step ${t.step||0} · ${d.checkpoint||'尚无 checkpoint'}`;$('steps').textContent=`${t.step||0} / ${t.max_steps||'—'}`;let percent=t.max_steps?100*t.step/t.max_steps:0;$('pct').textContent=`${percent.toFixed(2)}%`;$('bar').style.width=`${Math.min(100,percent)}%`;$('speed').textContent=t.speed?`${Number(t.speed).toFixed(2)} step/s`:'—';$('eta').textContent=t.speed&&t.max_steps?`预计还需 ${duration((t.max_steps-t.step)/t.speed)}`:'等待速度数据';$('loss').textContent=t.loss?Number(t.loss).toFixed(4):'—';$('lr').textContent=t.lr?`LR ${Number(t.lr).toExponential(2)}`:'—';let rows=[['GPU 利用率',g.utilization==null?'—':`${g.utilization}%`],['显存',g.memory||'—'],['温度 / 功耗',`${g.temperature||'—'}°C / ${g.power||'—'}W`],['最新 checkpoint',d.checkpoint||'—'],['更新时间',new Date(d.timestamp*1000).toLocaleTimeString()]];$('gpu').innerHTML=rows.map(x=>`<div class="metric"><span>${esc(x[0])}</span><b>${esc(x[1])}</b></div>`).join('');$('log').textContent=d.log||'等待训练日志…';$('runs').innerHTML=(d.runs||[]).map(x=>`<span class="run">${esc(x.name)} · ${esc(x.step)} steps · ${esc(x.status)}</span>`).join('');chart(d.history||[]);$('clock').textContent=new Date().toLocaleTimeString()}
async function refresh(){try{let r=await fetch('/api/status?t='+Date.now(),{cache:'no-store'});render(await r.json())}catch(e){$('state').textContent='OFFLINE';$('host').textContent=String(e)}}refresh();setInterval(refresh,2000);addEventListener('resize',()=>refresh());
</script></body></html>'''


def read_csv(path):
    try:
        with path.open(newline='', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    except (OSError, csv.Error):
        return []


def tail(path, lines=40):
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').splitlines()[-lines:])
    except OSError:
        return ''


def gpu_info():
    try:
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], text=True, stderr=subprocess.DEVNULL).splitlines()[0]
        u, used, total, temp, power = [x.strip() for x in raw.split(',')]
        return {'utilization': u, 'memory': f'{used} / {total} MiB', 'temperature': temp, 'power': power}
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return {}


def process_running(name):
    try:
        output = subprocess.check_output(['ps', '-eo', 'args='], text=True)
        return any('python' in line and 'projects/project4_flow_matching/train.py' in line and name in line for line in output.splitlines())
    except (OSError, subprocess.CalledProcessError):
        return False


def status(root):
    root = Path(root)
    storage_root = root
    run_root = root / 'runs'
    if not run_root.exists():
        run_root = root.parent / 'runs'
        storage_root = root.parent
    candidates = sorted(run_root.glob('fm_v2_*'))
    candidates += [run_root / 'fm'] if (run_root / 'fm').exists() else []
    run_cards, selected, selected_rows = [], None, []
    latest_mtime = -1.0
    for folder in candidates:
        rows = read_csv(folder / 'train_log.csv')
        last = rows[-1] if rows else {}
        running = process_running(folder.name.removeprefix('fm_v2_'))
        step = int(float(last.get('step') or 0))
        run_cards.append({'name': folder.name, 'step': step, 'status': '运行中' if running else ('已有记录' if step else '等待')})
        csv_path = folder / 'train_log.csv'
        if running:
            selected, selected_rows = folder, rows
            break
        if rows and csv_path.stat().st_mtime >= latest_mtime:
            latest_mtime = csv_path.stat().st_mtime
            selected, selected_rows = folder, rows
    selected = selected or run_root / 'fm_v2_unconditional'
    last = selected_rows[-1] if selected_rows else {}
    cfg_path = root / 'projects/project4_flow_matching/configs' / f'cifar10_{selected.name}.yaml'
    max_steps = 200000
    if cfg_path and cfg_path.exists():
        try:
            for line in cfg_path.read_text(encoding='utf-8').splitlines():
                if line.strip().startswith('max_steps:'):
                    max_steps = int(line.split(':',1)[1].strip())
        except (OSError, ValueError):
            pass
    running = process_running(selected.name.removeprefix('fm_v2_'))
    checkpoint = selected / 'latest.pt'
    checkpoint_text = f"latest.pt · {time.strftime('%H:%M:%S', time.localtime(checkpoint.stat().st_mtime))}" if checkpoint.exists() else None
    history = []
    for row in selected_rows:
        try:
            history.append({'step': int(row['step']), 'loss': float(row['loss_100'])})
        except (ValueError, KeyError):
            continue
    return {'state':'running' if running else ('training' if last else 'waiting'), 'stage':'正式训练' if running else ('已训练/等待启动' if last else '等待训练'), 'host':os.uname().nodename if hasattr(os,'uname') else 'AutoDL', 'run_dir':str(selected), 'runs':run_cards, 'training':{'step':int(float(last.get('step') or 0)), 'loss':last.get('loss_100'), 'lr':last.get('lr'), 'speed':last.get('step_per_sec'), 'elapsed':last.get('elapsed_seconds'), 'max_steps':max_steps}, 'history':history[-300:], 'gpu':gpu_info(), 'checkpoint':checkpoint_text, 'log':tail(storage_root / 'logs' / f'{selected.name}.log'), 'timestamp':time.time()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=6006)
    a = p.parse_args()
    root = Path(a.repo_root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if urlparse(self.path).path == '/api/status':
                payload = json.dumps(status(root), ensure_ascii=False).encode()
                self.send_response(200); self.send_header('Content-Type','application/json; charset=utf-8'); self.send_header('Cache-Control','no-store'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
            elif urlparse(self.path).path in ('/', '/index.html'):
                payload = HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.send_header('Content-Length',str(len(payload))); self.end_headers(); self.wfile.write(payload)
            else:
                self.send_error(404)
        def log_message(self, *_args):
            pass
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f'Project4 live dashboard at http://{a.host}:{a.port}/ ; reading {root}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
