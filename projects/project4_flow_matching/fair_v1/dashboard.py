"""Token-protected, read-only live monitor for Project4 fair-v1."""
from __future__ import annotations
import csv, hmac, json, os, re, secrets, subprocess, time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
ROOT = Path(__file__).resolve().parents[3]
CKPT_ROOT = Path(os.environ.get("P4_FAIR_CKPT_ROOT", "/root/autodl-tmp/checkpoints/project4-fair-v1"))
PORT = int(os.environ.get("P4_DASHBOARD_PORT", "8443"))
TOKEN_FILE = CKPT_ROOT / "dashboard.token"
_ALL_EXPERIMENTS = {"D-U": 200000, "F-U": 200000, "F-C": 200000, "F-LN": 200000,
                   "F-DiT": 200000, "F-CT": 50000, "F-RF": 50000}
_NODE_EXPERIMENTS = {
    "A": {"D-U": 200000, "F-U": 200000, "F-CT": 50000, "F-RF": 50000},
    "B": {"F-C": 200000, "F-LN": 200000},
    "C": {"F-DiT": 200000},
}
EXPERIMENTS = _NODE_EXPERIMENTS.get(os.environ.get("P4_DASHBOARD_NODE", "all"), _ALL_EXPERIMENTS)
if not TOKEN_FILE.exists():
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(secrets.token_urlsafe(24), encoding="ascii")
    TOKEN_FILE.chmod(0o600)
TOKEN = os.environ.get("P4_DASHBOARD_TOKEN") or TOKEN_FILE.read_text(encoding="ascii").strip()
HTML = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project4 fair-v1</title><style>
body{margin:24px;background:#0b1020;color:#e7eefc;font:15px system-ui}main{max-width:1100px;margin:auto}.muted{color:#9aabc7}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0}.card{background:#151d32;border:1px solid #293653;border-radius:12px;padding:14px}.value{font-size:21px;margin-top:6px}table{width:100%;border-collapse:collapse;background:#151d32}td,th{text-align:left;padding:10px;border-bottom:1px solid #293653}th{color:#9aabc7}.bar{height:6px;background:#293653;margin-top:6px}.bar i{display:block;height:100%;background:#50d890}.running{color:#50d890}.complete{color:#77a9ff}.stale,.failed{color:#ff7777}.blocked{color:#ffc857}.pending{color:#9aabc7}.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.step{min-height:112px}.step-detail{font-size:13px;line-height:1.4;margin-top:8px}.stepbar{height:5px;background:#293653;margin-top:8px}.stepbar i{display:block;height:100%;background:#50d890}@media(max-width:700px){.grid{grid-template-columns:repeat(2,1fr)}body{margin:12px}}
</style></head><body><main><h1>Project 4 · fair-v1</h1><div class="muted">AutoDL 实时实 · 5 秒刷新</div>
<div id="summary" class="grid"></div><div class="card"><h2>训练任务</h2><table><thead><tr><th>实验</th><th>状态 / 进度</th><th>吞吐</th><th>预计剩余</th><th>最近更新</th></tr></thead><tbody id="runs"></tbody></table></div><div class="card" style="margin-top:14px"><h2>评估、核验与备份</h2><div id="postprocess" class="steps"></div></div><p id="updated" class="muted"></p></main>
<script>
const token=new URLSearchParams(location.search).get('token');
function addText(parent,tag,text,cls){const el=document.createElement(tag);el.textContent=String(text);if(cls)el.className=cls;parent.appendChild(el);return el}
async function tick(){try{const r=await fetch('/api/state?token='+encodeURIComponent(token||''),{cache:'no-store'});if(!r.ok)throw Error(r.status);const d=await r.json();const summary=document.querySelector('#summary');summary.replaceChildren();for(const [label,value] of [['当前任务',d.current.experiment],['GPU',d.gpu.name],['GPU 利用率 / 显存',d.gpu.util+'% / '+d.gpu.memory],['状态 / 温度',d.current.status+' / '+d.gpu.temperature]]){const card=addText(summary,'div','', 'card');addText(card,'span',label,'muted');addText(card,'div',value,'value')}
const body=document.querySelector('#runs');body.replaceChildren();for(const x of d.runs){const tr=document.createElement('tr');const name=addText(tr,'td',x.id);const bar=addText(name,'div','','bar');const fill=document.createElement('i');fill.style.width=Math.min(100,100*x.step/x.max_steps)+'%';bar.appendChild(fill);addText(tr,'td',x.status+' · '+x.step.toLocaleString()+' / '+x.max_steps.toLocaleString(),x.status);addText(tr,'td',x.speed?x.speed.toFixed(3)+' step/s':'—');addText(tr,'td',x.eta_hours==null?'—':x.eta_hours.toFixed(1)+' h');addText(tr,'td',x.heartbeat);body.appendChild(tr)}const panel=document.querySelector('#postprocess');panel.replaceChildren();const labels={pending:'待开始',running:'进行中',complete:'完成',blocked:'受阻',failed:'失败',stale:'超时'};for(const stage of ((d.postprocess||{}).stages||[])){const card=addText(panel,'div','','card step');addText(card,'div',stage.label,'muted');addText(card,'div',labels[stage.status]||stage.status,stage.status);addText(card,'div',stage.detail||'—','step-detail');if(stage.progress!==undefined&&stage.progress!==null){const bar=addText(card,'div','','stepbar');const fill=document.createElement('i');fill.style.width=Math.max(0,Math.min(100,stage.progress))+'%';bar.appendChild(fill)}}document.querySelector('#updated').textContent='刷新 '+new Date(d.server_time).toLocaleString()+' · 心跳 '+d.current.heartbeat_age+'s'}catch(e){document.querySelector('#updated').textContent='连接异常：'+e.message}}
tick();setInterval(tick,5000);
</script></body></html>"""


def _json(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

def _gpu():
    cmd=["nvidia-smi","--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader,nounits"]
    try:
        raw=subprocess.check_output(cmd,text=True,timeout=2).strip().splitlines()[0]
        name,util,used,total,temp=[x.strip() for x in raw.split(",")]
        return {"name":name,"util":int(util),"memory":f"{int(used)/1024:.1f}/{int(total)/1024:.1f} GiB","temperature":f"{temp}°C"}
    except Exception: return {"name":"unavailable","util":"—","memory":"—","temperature":"—"}

def _last_row(path):
    if not path.exists(): return {}
    try:
        last=None
        with path.open(newline="",encoding="utf-8") as stream:
            for last in csv.DictReader(stream): pass
        return last or {}
    except Exception: return {}

def _postprocess(current):
    saved=_json(CKPT_ROOT/"node_c_postprocess_state.json")
    by_id={x.get("id"):x for x in saved.get("stages",[]) if isinstance(x,dict)}
    defaults=[
        ("evaluation","Euler 评估","待开始"),
        ("results","评估结果核验","待开始"),
        ("logs","日志核验","待开始"),
        ("checkpoint","Checkpoint 核验","待开始"),
        ("backup","持久化备份","待开始"),
    ]
    exp=current.get("experiment",""); status=current.get("status","")
    if exp=="eval-F-DiT-euler" and status=="running":
        log=ROOT/"results/fair_v1/F-DiT/euler.log"
        text=log.read_text(encoding="utf-8",errors="replace") if log.exists() else ""
        done=min(10,text.count("FID=")); nfe=re.findall(r"requested_nfe=(\d+)",text)
        by_id["evaluation"]={"id":"evaluation","label":"Euler 评估","status":"running",
            "progress":done*10,"detail":f"运行中：FID {done}/10 组；最近 NFE {nfe[-1] if nfe else '—'}"}
    elif exp=="node_C_complete" and "evaluation" not in by_id:
        by_id["evaluation"]={"id":"evaluation","label":"Euler 评估","status":"complete",
            "progress":100,"detail":"评估输出已生成，等待核验"}
    stages=[]
    for key,label,detail in defaults:
        stages.append(by_id.get(key,{"id":key,"label":label,"status":"pending","progress":0,"detail":detail}))
    return {"updated_at":saved.get("updated_at"),"stages":stages}


def _state():
    current=_json(CKPT_ROOT/"current_state.json")
    current_exp=current.get("experiment","idle"); current_status=current.get("status","idle")
    now=time.time(); runs=[]; active_age=None
    for experiment,maximum in EXPERIMENTS.items():
        output=CKPT_ROOT/experiment; row=_last_row(output/"train_log.csv")
        step=int(float(row.get("step",0) or 0)); speed=float(row.get("step_per_sec",0) or 0)
        eta=max(0,(maximum-step)/speed/3600) if speed>0 else None
        status="queued"
        if experiment==current_exp: status=current_status
        elif step>=maximum: status="complete"
        elif (output/"latest.pt").exists(): status="paused"
        signal=output/"train_log.csv"
        if not signal.exists(): signal=output/"latest.pt"
        age=int(now-signal.stat().st_mtime) if signal.exists() else None
        if experiment==current_exp: active_age=age
        if status=="running" and age is not None and age>1200: status="stale"
        runs.append({"id":experiment,"status":status,"step":step,"max_steps":maximum,
                     "speed":speed or None,"eta_hours":eta,"heartbeat":f"{age}s ago" if age is not None else "waiting"})
    heartbeat_age=active_age
    gpu=_gpu(); gpu.setdefault("memory_total","—")
    return {"current":{"experiment":current_exp,"status":current_status,"heartbeat_age":heartbeat_age if heartbeat_age is not None else "—"},
            "gpu":gpu,"runs":runs,"postprocess":_postprocess(current),"server_time":datetime.now(timezone.utc).isoformat()}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args): return
    def do_GET(self):
        parsed=urlparse(self.path); supplied=parse_qs(parsed.query).get("token",[""])[0]
        if not hmac.compare_digest(supplied,TOKEN):
            self.send_error(401,"dashboard token required"); return
        if parsed.path=="/api/state":
            payload=json.dumps(_state(),ensure_ascii=False).encode("utf-8"); content="application/json; charset=utf-8"
        elif parsed.path=="/":
            payload=HTML.encode("utf-8"); content="text/html; charset=utf-8"
        else:
            self.send_error(404); return
        self.send_response(200); self.send_header("Content-Type",content)
        self.send_header("Content-Length",str(len(payload))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(payload)

if __name__=="__main__":
    print(f"Project4 fair-v1 dashboard listening on 0.0.0.0:{PORT}",flush=True)
    print(f"access_token={TOKEN}",flush=True)
    ThreadingHTTPServer(("0.0.0.0",PORT),Handler).serve_forever()
