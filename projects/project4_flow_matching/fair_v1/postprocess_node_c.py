#!/usr/bin/env python3
"""Validate F-DiT outputs and archive them to AutoDL file storage if mounted."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tarfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CKPT = Path(os.environ.get("P4_FAIR_CKPT_ROOT", "/root/autodl-tmp/checkpoints/project4-fair-v1"))
STATE = CKPT / "node_c_postprocess_state.json"
RUN = CKPT / "F-DiT"
RESULTS = ROOT / "results/fair_v1/F-DiT"
RESULT_JSON = RESULTS / "euler.json"
RESULT_LOG = RESULTS / "euler.log"
MODEL = RUN / "step_200000.pt"
NOISE = ROOT / "results/project4_fair_v1/noise/noise_seed1001_n5000.pt"
NFE = [2, 4, 8, 10, 16, 20, 32, 50, 64, 100]
STAGES = [
    {"id": "evaluation", "label": "Euler 评估", "status": "pending", "progress": 0, "detail": "等待评估输出"},
    {"id": "results", "label": "评估结果核验", "status": "pending", "progress": 0, "detail": "等待核验"},
    {"id": "logs", "label": "日志核验", "status": "pending", "progress": 0, "detail": "等待核验"},
    {"id": "checkpoint", "label": "Checkpoint 核验", "status": "pending", "progress": 0, "detail": "等待核验"},
    {"id": "backup", "label": "持久化备份", "status": "pending", "progress": 0, "detail": "等待文件存储"},
]


def publish() -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    doc = {"node": "C", "experiment": "F-DiT", "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "stages": STAGES}
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(STATE)


def set_stage(key: str, status: str, detail: str, progress: int | None = None) -> None:
    for stage in STAGES:
        if stage["id"] == key:
            stage.update(status=status, detail=detail)
            if progress is not None:
                stage["progress"] = max(0, min(100, int(progress)))
            break
    publish()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> tuple[dict, str, str]:
    set_stage("evaluation", "complete", "Euler NFE 2–100；5,000 样本；评估 JSON 已生成", 100)
    set_stage("results", "running", "检查 NFE、FID、样本数及固定噪声库")
    result = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    rows = result.get("results", [])
    nfe = [int(row.get("effective_nfe_per_sample", -1)) for row in rows]
    if nfe != NFE or result.get("num_samples") != 5000 or result.get("generation_seed") != 1001 or int(result.get("checkpoint_step", -1)) != 200000:
        raise ValueError(f"评估字段不匹配：NFE={nfe}, samples={result.get('num_samples')}, step={result.get('checkpoint_step')}")
    if not all(math.isfinite(float(row["fid"])) for row in rows):
        raise ValueError("FID 缺失或非有限")
    if not NOISE.is_file():
        raise FileNotFoundError(f"固定噪声库缺失：{NOISE}")
    noise_hash = sha256(NOISE)
    if noise_hash != result.get("noise_bank_sha256"):
        raise ValueError("固定噪声库 SHA-256 不匹配")
    set_stage("results", "complete", "10 个 NFE 点、FID、样本数和固定噪声库校验通过", 100)

    set_stage("logs", "running", "检查评估日志正常结束及异常")
    if not RESULT_LOG.is_file() or RESULT_LOG.stat().st_size == 0:
        raise FileNotFoundError("评估日志不存在或为空")
    log = RESULT_LOG.read_text(encoding="utf-8", errors="replace")
    if "Traceback (most recent call last)" in log or "saved " + str(RESULT_JSON) not in log or log.count("FID=") != len(NFE):
        raise ValueError("评估日志不完整或包含异常")
    set_stage("logs", "complete", "正常结束；10 组 FID；无 Traceback", 100)

    set_stage("checkpoint", "running", "校验最终 checkpoint、训练步数与评估记录")
    if not MODEL.is_file() or MODEL.stat().st_size < 100 * 1024 * 1024:
        raise FileNotFoundError("最终 checkpoint 缺失或大小异常")
    checkpoint_hash = sha256(MODEL)
    if checkpoint_hash != result.get("checkpoint_sha256"):
        raise ValueError("checkpoint SHA-256 与评估记录不一致")
    last_train_row = (RUN / "train_log.csv").read_text(encoding="utf-8", errors="replace").splitlines()[-1]
    if not last_train_row.startswith("200000,"):
        raise ValueError("训练日志最后一步不是 200000")
    set_stage("checkpoint", "complete", f"{MODEL.stat().st_size / 1024**2:.1f} MiB；SHA-256 一致；训练到 200000 step", 100)
    return result, noise_hash, checkpoint_hash


def archive(result: dict, noise_hash: str, checkpoint_hash: str) -> None:
    storage = Path("/root/autodl-fs")
    if not storage.is_dir() or not os.path.ismount(storage):
        set_stage("backup", "blocked", "AutoDL 文件存储未挂载（/root/autodl-fs）；/root/autodl-pub 为只读盘", 0)
        return
    destination = storage / "Project4" / "fair_v1" / "node_C"
    try:
        destination.mkdir(parents=True, exist_ok=True)
        probe = destination / f".write_probe_{os.getpid()}"
        probe.write_text("ok", encoding="ascii")
        probe.unlink()
    except Exception as exc:
        set_stage("backup", "blocked", f"文件存储不可写：{exc}", 0)
        return

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output = destination / f"F-DiT_seed44_{stamp}.tar"
    partial = output.with_suffix(".tar.partial")
    inputs = [
        (ROOT / "projects/project4_flow_matching", "source/projects/project4_flow_matching"),
        (RUN / "step_200000.pt", "checkpoint/step_200000.pt"),
        (RUN / "run_metadata.json", "checkpoint/run_metadata.json"),
        (RUN / "train_log.csv", "checkpoint/train_log.csv"),
        (RUN / "console.log", "checkpoint/console.log"),
        (RESULT_JSON, "evaluation/euler.json"),
        (RESULT_LOG, "evaluation/euler.log"),
        (RESULTS / "euler", "evaluation/euler_samples"),
        (NOISE, "evaluation/noise_seed1001_n5000.pt"),
    ]
    inputs = [(path, name) for path, name in inputs if path.exists()]
    set_stage("backup", "running", f"归档到 AutoDL 文件存储：{output}", 1)
    try:
        with tarfile.open(partial, "w") as tar:
            for index, (path, arcname) in enumerate(inputs, 1):
                tar.add(path, arcname=arcname, recursive=True)
                set_stage("backup", "running", f"归档 {index}/{len(inputs)}：{path.name}", 100 * index // len(inputs))
            manifest = {"experiment": "F-DiT", "node": "C", "training_seed": 44, "solver": "euler", "nfe": NFE,
                        "num_samples": 5000, "generation_seed": 1001, "git_revision": result.get("git_revision"),
                        "checkpoint_sha256": checkpoint_hash, "noise_bank_sha256": noise_hash}
            payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        partial.replace(output)
        archive_hash = sha256(output)
        output.with_suffix(".tar.sha256").write_text(f"{archive_hash}  {output.name}\n", encoding="ascii")
        set_stage("backup", "complete", f"归档完成：{output}；SHA-256 {archive_hash[:16]}…", 100)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        set_stage("backup", "failed", f"归档失败：{exc}", 0)


def main() -> None:
    current_file = CKPT / "current_state.json"
    current = json.loads(current_file.read_text(encoding="utf-8")) if current_file.exists() else {}
    if current.get("status") == "running" and str(current.get("experiment", "")).startswith("eval-"):
        log = RESULT_LOG.read_text(encoding="utf-8", errors="replace") if RESULT_LOG.exists() else ""
        finished = min(len(NFE), log.count("FID="))
        set_stage("evaluation", "running", f"Euler 评估中：已完成 {finished}/{len(NFE)} 组", finished * 100 // len(NFE))
        return
    result, noise_hash, checkpoint_hash = verify()
    set_stage("backup", "running", "检查 AutoDL 文件存储挂载状态")
    archive(result, noise_hash, checkpoint_hash)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        for stage in STAGES:
            if stage["status"] == "running":
                stage.update(status="failed", detail=str(exc), progress=0)
        publish()
        raise
