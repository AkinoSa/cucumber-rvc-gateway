#!/usr/bin/env python
"""RVC 常驻推理 worker（CUSTOM-7.2）。

为什么存在：网关原本每次推理都新起 `infer/cli.py` 子进程，模型/HuBERT/RMVPE
每次都要重新加载（i5-6500 CPU 上固定 ~40s），TTS 这类短音频的加载成本占比极高。
本进程常驻、保持模型在内存，单次推理只剩实际计算（短音频 3~5s）。

协议（stdin/stdout，JSON Lines）：
    请求: {"id": "...", "model": "rym.pth", "input": "...", "output": "...",
           "pitch": 0, "f0_method": "rmvpe", "index": "/path/x.index",
           "index_rate": 0.75, "resample_sr": 0, "rms_mix_rate": 0.35,
           "protect": 0.33, "output_format": "wav", "speaker_id": 0}
    响应: {"id": "...", "ok": true, "elapsed": 3.2} 或 {"id": "...", "ok": false, "error": "..."}

约定：
    * 协议走**复制的 stdout**；进程真正的 stdout 被重定向到 stderr，
      防止 torch/transformers 的杂散 print 破坏 JSON 流。
    * 空闲 IDLE 秒后自动退出（网关检测到 EOF 会在下次请求时重新拉起），
      避免模型长期占用内存。
    * 推理/加载逻辑全部复用 /app/infer/cli.py 的现成函数，行为与子进程完全一致。
"""

import importlib.util
import json
import os
import select
import sys
import time

APP_DIR = os.environ.get("GW_APP_DIR", "/app")
WEIGHTS_DIR = os.environ.get("GW_WEIGHTS_DIR", os.path.join(APP_DIR, "assets", "weights"))
IDLE_EXIT_SECONDS = int(os.environ.get("GW_WORKER_IDLE_SECONDS", "600"))

os.chdir(APP_DIR)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ["weight_root"] = WEIGHTS_DIR
os.environ.setdefault("outside_index_root", os.path.join(APP_DIR, "assets", "indices"))
os.environ.setdefault("rmvpe_root", os.path.join(APP_DIR, "assets", "rmvpe"))

# 协议通道：复制原 stdout；之后所有杂散输出都进 stderr
_protocol = os.fdopen(os.dup(1), "w", buffering=1)
sys.stdout = sys.stderr


def send(payload: dict) -> None:
    _protocol.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _protocol.flush()


def load_cli_module():
    spec = importlib.util.spec_from_file_location(
        "rvc_cli", os.path.join(APP_DIR, "infer", "cli.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    cli = load_cli_module()
    from infer.vc.modules import VC

    config = cli.create_config()
    vc = None
    cur_model = None
    cur_speaker = None
    send({"ok": True, "event": "ready"})

    stdin = sys.stdin.buffer
    while True:
        ready, _, _ = select.select([stdin], [], [], IDLE_EXIT_SECONDS)
        if not ready:
            send({"ok": True, "event": "idle_exit", "idle": IDLE_EXIT_SECONDS})
            return 0
        line = stdin.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception as exc:
            send({"id": None, "ok": False, "error": f"bad request: {exc}"})
            continue

        rid = req.get("id")
        started = time.monotonic()
        try:
            model_name = str(req.get("model") or "")
            model_path = cli.resolve_model(model_name)
            os.environ["weight_root"] = str(model_path.parent)
            speaker_count, speakers = cli.load_model_metadata(model_path)
            speaker_id = cli.select_speaker(speaker_count, speakers, req.get("speaker_id"))

            index_rate = float(req.get("index_rate", 0.0) or 0.0)
            index_path = cli.resolve_index(
                str(req.get("index") or ""), model_name, speaker_id, index_rate)

            # 模型变化才重新加载（同模型连续请求 0 加载成本）
            if vc is None or cur_model != model_name:
                t0 = time.monotonic()
                vc = VC(config)
                vc.get_vc(model_path.name)
                cur_model = model_name
                print("[worker] 模型已加载: %s（%.1fs）" % (model_name, time.monotonic() - t0))

            status, result = vc.vc_single(
                speaker_id,
                str(req.get("input")),
                int(req.get("pitch", 0) or 0),
                str(req.get("f0_method") or "rmvpe"),
                index_path,
                index_rate,
                int(req.get("resample_sr", 0) or 0),
                float(req.get("rms_mix_rate", 1.0) or 1.0),
                float(req.get("protect", 0.33) or 0.0),
            )
            if not result or result[0] is None or result[1] is None:
                raise RuntimeError("推理未产出音频: %s" % status)
            cli.write_audio(
                __import__("pathlib").Path(str(req.get("output"))),
                result[1], result[0], str(req.get("output_format") or "wav"))
            send({"id": rid, "ok": True,
                  "elapsed": round(time.monotonic() - started, 2),
                  "model_loaded": cur_model == model_name,
                  "status": str(status)[:200]})
        except Exception as exc:  # noqa: BLE001 — 单请求失败不退出，保持 worker 存活
            send({"id": rid, "ok": False,
                  "error": "%s: %s" % (type(exc).__name__, exc)})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
