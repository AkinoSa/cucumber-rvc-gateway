#!/usr/bin/env python
"""RVC 控制台网关（cucumber-rvc-gateway）

独立中间层：接收插件的 Gradio 调用，完成
  取音源 -> 探测时长 -> 校验裁剪参数 -> FFmpeg 裁剪
  -> 3HP 人声分离(PyMSS) -> RVC 音色转换(infer/cli.py) -> 返回宿主机可读取路径

设计约束：
  * Gradio 服务由 /opt/gwvenv/bin/python 运行（隔离 huggingface-hub 升级）
  * 所有推理子进程一律用系统 Python /usr/local/bin/python（保证 transformers/HuBERT 原样）
  * 不使用 shell 拼接，全部用参数数组启动子进程
  * 全局串行：Gradio 队列 + 进程内信号量；任务按 task_id 隔离
"""

from __future__ import annotations

import errno
import atexit
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import gradio as gr

# --------------------------------------------------------------------------- #
# 配置（全部来自环境变量，便于后续调整而不重建镜像）
# --------------------------------------------------------------------------- #

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


SYSTEM_PYTHON = _env("GW_SYSTEM_PYTHON", "/usr/local/bin/python")
APP_DIR = _env("GW_APP_DIR", "/app")
WEIGHTS_DIR = _env("GW_WEIGHTS_DIR", "/app/assets/weights")
INDEX_DIR = _env("GW_INDEX_DIR", "/app/assets/indices")
PYMSS_MODEL_DIR = _env("GW_PYMSS_MODEL_DIR", "/app/assets/pymss_weights")

CONTAINER_WORK_DIR = _env("GW_CONTAINER_WORK_DIR", "/work")
CONTAINER_PLUGIN_DIR = _env("GW_CONTAINER_PLUGIN_DIR", "/mnt/plugin_data")
# CUSTOM-9 通用化：宿主机路径一律由环境变量提供（compose/.env），不再写死默认值
HOST_WORK_DIR = _env("GW_HOST_WORK_DIR", "")
HOST_PLUGIN_DIR = _env("GW_HOST_PLUGIN_DIR", "")

MAX_CLIP_SECONDS = _env_float("GW_MAX_CLIP_SECONDS", 30.0)
# chorus 模式：送去分离的候选片段 = 用户目标时长 + 这段缓冲（分离后的边界更干净），
# 分离与推理完成后再在候选片段内精挑最终输出区间。CUSTOM-4 起由 15 秒改为 5 秒。
SEP_BUFFER_SECONDS = _env_float("GW_SEP_BUFFER_SECONDS", 5.0)
# 人声静音判定：分离后 vocals 的平均电平低于该值，视为“这段没人唱”，直接报错
VOICE_SILENT_DB = _env_float("GW_VOICE_SILENT_DB", -50.0)
# 人声“偏轻”阈值：低于它时换段再试，最后取各次尝试里人声最响的一段
QUIET_VOICE_DB = _env_float("GW_QUIET_VOICE_DB", -35.0)
# 换段重试次数（0=不重试）
SEP_SILENT_RETRY = _env_int("GW_SEP_SILENT_RETRY", 2)
# 最终长度选择：每多 1 秒的轻微偏好惩罚（0=纯按得分，越大越偏好短片段）
LENGTH_PREFERENCE = _env_float("GW_LENGTH_PREFERENCE", 0.03)
DEFAULT_CLIP_SECONDS = _env_float("GW_DEFAULT_CLIP_SECONDS", 30.0)
MAX_INPUT_SECONDS = _env_float("GW_MAX_INPUT_SECONDS", 900.0)
MAX_INPUT_BYTES = _env_int("GW_MAX_INPUT_BYTES", 80 * 1024 * 1024)

SEP_MODEL_DEFAULT = _env("GW_SEP_MODEL", "3_HP-Vocal-UVR.pth")

DEFAULT_PITCH = _env_int("GW_DEFAULT_PITCH", 0)
DEFAULT_F0_METHOD = _env("GW_DEFAULT_F0", "rmvpe")
DEFAULT_INDEX_RATE = _env_float("GW_DEFAULT_INDEX_RATE", 0.75)
DEFAULT_PROTECT = _env_float("GW_DEFAULT_PROTECT", 0.33)
DEFAULT_RESAMPLE_SR = _env_int("GW_DEFAULT_RESAMPLE_SR", 0)
DEFAULT_RMS_MIX_RATE = _env_float("GW_DEFAULT_RMS_MIX_RATE", 1.0)
DEFAULT_OUTPUT_FORMAT = _env("GW_DEFAULT_FORMAT", "wav")

FFMPEG_TIMEOUT = _env_int("GW_FFMPEG_TIMEOUT", 180)
SEP_TIMEOUT = _env_int("GW_SEP_TIMEOUT", 1200)
RVC_TIMEOUT = _env_int("GW_RVC_TIMEOUT", 1200)
DOWNLOAD_TIMEOUT = _env_int("GW_DOWNLOAD_TIMEOUT", 90)
QUEUE_MAX_SIZE = _env_int("GW_QUEUE_MAX", 4)

NETEASE_BASE = _env("GW_NETEASE_BASE", "https://wyy.xhily.com")

ALLOWED_FORMATS = ("wav", "flac", "mp3", "m4a")
ALLOWED_F0 = ("pm", "rmvpe")
# 推理线程数：i5-6500 只有 4 线程，线程越多峰值内存越高。
# 实测 30s 片段在 4 线程下会被 OOM kill，故默认压到 2。
OMP_THREADS = _env("GW_OMP_THREADS", "2")

# ---- 常驻 RVC worker（CUSTOM-7.2）：避免每次推理重新加载模型（CPU 上固定 ~40s）----
WORKER_ENABLED = _env_int("GW_WORKER", 1)
WORKER_IDLE_SECONDS = _env_int("GW_WORKER_IDLE_SECONDS", 600)  # 空闲自动退出，省内存

# ---- 副歌挑选（轻量级初版，见 pick_chorus_window 文档）----
CHORUS_ANALYSIS_SR = _env_int("GW_CHORUS_SR", 22050)      # 包络分析用的采样率
CHORUS_HOP_SECONDS = _env_float("GW_CHORUS_HOP", 0.1)     # 包络帧长
CHORUS_STEP_SECONDS = _env_float("GW_CHORUS_STEP", 0.5)   # 滑窗步长
CHORUS_SNAP_WINDOW = _env_float("GW_CHORUS_SNAP", 2.0)    # 边界吸附搜索半径（秒）
CHORUS_MIN_SECONDS = _env_float("GW_CHORUS_MIN", 5.0)     # 太短的音频直接取整段

logger = logging.getLogger("rvc-gateway")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# --------------------------------------------------------------------------- #
# 任务控制：全局串行 + 子进程登记（用于取消）
# --------------------------------------------------------------------------- #

_TASK_SLOT = threading.Semaphore(1)
_ACTIVE_PROCS: dict[str, list[subprocess.Popen]] = {}
_PROC_LOCK = threading.Lock()


class GatewayError(Exception):
    """面向用户的可预期错误。"""


def _register_proc(task_id: str, proc: subprocess.Popen) -> None:
    with _PROC_LOCK:
        _ACTIVE_PROCS.setdefault(task_id, []).append(proc)


def _kill_procs(task_id: str) -> int:
    """终止某任务登记的所有子进程，返回实际发出终止信号的进程数。"""
    killed = 0
    with _PROC_LOCK:
        procs = _ACTIVE_PROCS.pop(task_id, [])
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.terminate()
                killed += 1
            except Exception:  # noqa: BLE001
                pass
    for proc in procs:
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
    return killed


def _cleanup_task_dir(task_dir: Path) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


def sweep_stale_tasks(ttl_seconds: int = 3600) -> int:
    """清理超时的历史任务目录（例如网关重启打断清理时留下的残骸）。

    只处理 TASK_ROOT 下超过 ttl 的目录，绝不触碰模型、音源、输出目录。
    """
    if not TASK_ROOT.is_dir():
        return 0
    removed = 0
    now = time.time()
    for d in TASK_ROOT.iterdir():
        if not d.is_dir():
            continue
        try:
            if now - d.stat().st_mtime > ttl_seconds:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info("清理陈旧任务目录 %d 个（TTL=%ds）", removed, ttl_seconds)
    return removed


# --------------------------------------------------------------------------- #
# 模型 / 索引 动态发现
# --------------------------------------------------------------------------- #

def list_rvc_models() -> list[str]:
    """动态扫描音色目录，返回 .pth 文件名列表。"""
    base = Path(WEIGHTS_DIR)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.glob("*.pth") if p.is_file())


def find_index_for(model_name: str) -> str | None:
    """按模型名寻找配对 index：精确 <stem>.index 优先，其次文件名包含 stem。"""
    stem = Path(str(model_name)).stem.lower()
    base = Path(INDEX_DIR)
    if not base.is_dir():
        return None
    exact = base / f"{stem}.index"
    if exact.is_file():
        return str(exact)
    candidates = sorted(p for p in base.glob("*.index") if stem in p.name.lower())
    return str(candidates[0]) if candidates else None


def list_sep_models() -> list[dict]:
    """扫描分离模型目录，返回插件需要的 {id, name} 结构（id 用于 select/传参）。"""
    base = Path(PYMSS_MODEL_DIR)
    out: list[dict] = []
    if base.is_dir():
        for p in sorted(base.rglob("*.pth")):
            out.append({"id": p.name, "name": p.stem})
    return out


def sep_model_path(model_id: str) -> Path:
    base = Path(PYMSS_MODEL_DIR)
    direct = base / model_id
    if direct.is_file():
        return direct
    for p in sorted(base.rglob("*.pth")):
        if p.name == model_id:
            return p
    raise GatewayError(f"分离模型不存在或未挂载: {model_id}")


# --------------------------------------------------------------------------- #
# 路径：容器 <-> 宿主机
# --------------------------------------------------------------------------- #

def to_container_path(raw: str) -> str:
    """把插件给的宿主机路径翻译成容器内路径；不识别则原样返回。"""
    if not raw:
        return raw
    raw = raw.strip().strip('"').strip("'")
    mappings = (
        (HOST_WORK_DIR, CONTAINER_WORK_DIR),
        (HOST_PLUGIN_DIR, CONTAINER_PLUGIN_DIR),
    )
    for host_prefix, ct_prefix in mappings:
        if not host_prefix:
            continue  # 未配置的映射跳过（容器内路径原样返回）
        if raw.startswith(host_prefix.rstrip("/") + "/"):
            return ct_prefix.rstrip("/") + raw[len(host_prefix.rstrip("/")):]
    return raw


def to_host_path(container_path: str) -> str:
    raw = str(container_path)
    if raw.startswith(CONTAINER_WORK_DIR.rstrip("/") + "/"):
        return HOST_WORK_DIR.rstrip("/") + raw[len(CONTAINER_WORK_DIR.rstrip("/")):]
    if raw.startswith(CONTAINER_PLUGIN_DIR.rstrip("/") + "/"):
        return HOST_PLUGIN_DIR.rstrip("/") + raw[len(CONTAINER_PLUGIN_DIR.rstrip("/")):]
    return raw


# --------------------------------------------------------------------------- #
# 音源获取
# --------------------------------------------------------------------------- #

def _download(url: str, dest: Path) -> None:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "rvc-gateway/1.0"})
    with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
        data = resp.read(MAX_INPUT_BYTES + 1)
    if len(data) > MAX_INPUT_BYTES:
        raise GatewayError("音源文件过大，已拒绝下载")
    dest.write_bytes(data)
    if not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError("音源下载结果为空")


def _fetch_netease_url(song_id: str) -> str:
    """按插件 nodejs 音源的 /song/url 接口取播放链接（网络不可达时会明确报错）。"""
    import urllib.parse
    import urllib.request

    url = NETEASE_BASE.rstrip("/") + "/song/url"
    body = urllib.parse.urlencode({"id": song_id}).encode()
    req = urllib.request.Request(url, data=body)
    try:
        with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:  # noqa: S310
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:  # noqa: BLE001
        raise GatewayError(
            f"网易云音源接口不可达（{type(exc).__name__}: {exc}）。"
            "请改用插件侧已下载的本地音频（QQ音乐源）或提供 http(s) 直链。"
        ) from exc
    data = (payload or {}).get("data") or []
    if not data or not data[0].get("url"):
        raise GatewayError(f"网易云未返回可用播放链接（id={song_id}）")
    return data[0]["url"]


def resolve_source(raw: str, task_dir: Path) -> Path:
    """返回容器内可直接读取的音频文件路径。"""
    if not raw or not str(raw).strip():
        raise GatewayError("song_name_src 为空")
    raw = str(raw).strip()

    if raw.startswith("http://") or raw.startswith("https://"):
        dest = task_dir / f"source{_guess_suffix(raw)}"
        _download(raw, dest)
        return dest

    candidate = to_container_path(raw)
    p = Path(candidate)
    if p.is_file():
        return p

    if re.fullmatch(r"\d{3,}", raw):
        url = _fetch_netease_url(raw)
        dest = task_dir / f"source{_guess_suffix(url)}"
        _download(url, dest)
        return dest

    raise GatewayError(
        f"无法读取音源: {raw!r}（容器内解析为 {candidate!r}）。"
        "请确认该文件已存在于已挂载目录，或改用 http(s) 直链 / 网易云歌曲 ID。"
    )


def _guess_suffix(url: str) -> str:
    m = re.search(r"\.(mp3|wav|flac|m4a|ogg|aac)(?:\?|$)", url.lower())
    return "." + m.group(1) if m else ".mp3"


# --------------------------------------------------------------------------- #
# 音频工具
# --------------------------------------------------------------------------- #

def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        raise GatewayError(f"无法探测音频时长: {proc.stderr.strip()[:200]}")
    try:
        value = float(proc.stdout.strip().splitlines()[0])
    except Exception as exc:  # noqa: BLE001
        raise GatewayError(f"音频时长解析失败: {proc.stdout[:120]!r}") from exc
    if value <= 0:
        raise GatewayError(f"音频时长非法: {value}")
    return value


def convert_audio_fmt(src: Path, dest: Path, fmt: str) -> None:
    """把 wav 转成目标格式；wav 直接复制以省 CPU。"""
    fmt = (fmt or "wav").lower()
    if fmt == "wav":
        import shutil
        shutil.copy2(src, dest)
        return
    codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
             "flac": ["-c:a", "flac"],
             "m4a": ["-c:a", "aac", "-b:a", "320k"]}.get(fmt)
    if not codec:
        raise GatewayError(f"不支持的导出格式: {fmt}")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), *codec, str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError(f"转码失败({fmt}): {proc.stderr.strip()[:300]}")


def clip_audio(song_name_src="", seconds=25, output_format="wav",
               progress=gr.Progress()):
    """只切段：用与翻唱相同的启发式在音源里挑目标时长的片段并导出，不做分离/变声。"""
    import shutil

    try:
        seconds = int(seconds or 25)
    except Exception:
        seconds = 25
    seconds = max(10, min(MAX_CLIP_SECONDS, seconds))
    fmt = str(output_format or "wav").strip().lower()
    if fmt not in ("wav", "mp3", "flac", "m4a"):
        fmt = "wav"

    task_id = uuid.uuid4().hex[:12]
    task_dir = TASK_ROOT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        progress(0.1, desc="获取音源")
        src_path = resolve_source(song_name_src, task_dir)
        size = src_path.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise GatewayError(f"音源文件过大: {size / 1048576:.1f}MB（上限 {MAX_INPUT_BYTES / 1048576:.0f}MB）")
        total = probe_duration(src_path)
        if total > MAX_INPUT_SECONDS:
            raise GatewayError(f"音源过长: {total:.1f}s（上限 {MAX_INPUT_SECONDS:.0f}s）")
        target = float(min(seconds, total))
        progress(0.4, desc="挑选片段")
        start, length, clip_source = pick_chorus_window(total, target, src_path, task_id)
        if start < 0 or length <= 0 or start >= total:
            raise GatewayError("挑选片段失败（音源太短或无法分析）")
        length = min(length, MAX_CLIP_SECONDS, total - start)
        clip = task_dir / "clip.wav"
        ffmpeg_trim(src_path, start, length, clip)
        dest = OUT_DIR / f"{task_id}_clip.{fmt}"
        convert_audio_fmt(clip, dest, fmt)
        os.chmod(dest, 0o644)
        logger.info("[%s] 切段完成: %.2fs~%.2fs（%.2fs，来源 %s）",
                    task_id, start, start + length, length, clip_source)
        progress(1.0, desc="完成")
        return {
            "task_id": task_id,
            "start": round(start, 3),
            "length": round(length, 3),
            "clip_source": clip_source,
            "source_duration": round(total, 3),
            "format": fmt,
            "elapsed": round(time.monotonic() - started, 2),
            "file": {
                "name": dest.name,
                "host_path": to_host_path(str(dest)),
                "size": dest.stat().st_size,
                "duration": round(probe_duration(dest), 3),
            },
        }
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


def separate_audio(song_name_src="", msst_model="", output_format="wav",
                   mode="both", progress=gr.Progress()):
    """只做人声/伴奏分离，不跑 RVC。mode: both / vocals / instrumental。"""
    import shutil

    mode = str(mode or "both").strip().lower()
    if mode not in ("both", "vocals", "instrumental"):
        mode = "both"
    fmt = str(output_format or "wav").strip().lower()
    if fmt not in ("wav", "mp3", "flac", "m4a"):
        fmt = "wav"

    task_id = uuid.uuid4().hex[:12]
    task_dir = TASK_ROOT / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        progress(0.05, desc="获取音源")
        src_path = resolve_source(song_name_src, task_dir)
        size = src_path.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise GatewayError(f"音源文件过大: {size / 1048576:.1f}MB（上限 {MAX_INPUT_BYTES / 1048576:.0f}MB）")
        total = probe_duration(src_path)
        if total > MAX_INPUT_SECONDS:
            raise GatewayError(f"音源过长: {total:.1f}s（上限 {MAX_INPUT_SECONDS:.0f}s）")

        sep_model = str(msst_model or "").strip() or SEP_MODEL_DEFAULT
        try:
            sep_model_path(sep_model)
        except GatewayError as exc:
            logger.warning("[%s] 分离模型 %s 不可用（%s），回退默认 %s",
                           task_id, sep_model, exc, SEP_MODEL_DEFAULT)
            sep_model = SEP_MODEL_DEFAULT
            sep_model_path(sep_model)

        sep_dir = task_dir / "sep"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(APP_DIR, "tools")
        env["PYMSS_MODEL_DIR"] = PYMSS_MODEL_DIR
        env["OMP_NUM_THREADS"] = OMP_THREADS
        cmd = [SYSTEM_PYTHON, "-m", "pymss.cli", "infer", sep_model,
               "-i", str(src_path), "-o", str(sep_dir),
               "--device", "cpu", "--format", "wav", "--param", "batch_size=1"]
        logger.info("[%s] 分离开始: %s（模型 %s，时长 %.2fs）", task_id, src_path.name, sep_model, total)
        progress(0.25, desc="人声分离")
        proc = subprocess.run(cmd, cwd=APP_DIR, env=env, capture_output=True,
                              text=True, timeout=SEP_TIMEOUT)
        if proc.returncode != 0:
            raise GatewayError(f"分离失败: {proc.stderr.strip()[-400:]}")
        vocals = sorted(sep_dir.rglob("*_Vocals.wav"))
        insts = sorted(sep_dir.rglob("*_Instrumental.wav"))
        if not vocals and not insts:
            raise GatewayError("分离未产出结果文件")

        result = {"task_id": task_id, "mode": mode, "sep_model": sep_model,
                  "format": fmt, "source_duration": round(total, 3),
                  "vocals": None, "instrumental": None}
        for label, files in (("vocals", vocals), ("instrumental", insts)):
            if mode != "both" and mode != label:
                continue
            if not files:
                continue
            dest = OUT_DIR / f"{task_id}_{label}.{fmt}"
            convert_audio_fmt(files[0], dest, fmt)
            os.chmod(dest, 0o644)
            result[label] = {
                "name": dest.name,
                "host_path": to_host_path(str(dest)),
                "size": dest.stat().st_size,
                "duration": round(probe_duration(dest), 3),
            }
        if result["vocals"] is None and result["instrumental"] is None:
            raise GatewayError("分离结果为空")
        result["elapsed"] = round(time.monotonic() - started, 2)
        logger.info("[%s] 分离完成: %s（%.1fs）", task_id,
                    ", ".join(k for k in ("vocals", "instrumental") if result.get(k)),
                    result["elapsed"])
        progress(1.0, desc="完成")
        return result
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


# --- 混响（CUSTOM-7）：真正的卷积混响（合成 IR + FFT 卷积），只处理 RVC 输出人声 ---
# 8 个参数全部真实生效，不存在「界面上有但不起作用」的假参数。
REVERB_WET_DEFAULT = _env_float("GW_REVERB_WET", 0.18)        # 湿声比例 0~0.5
REVERB_DRY_DEFAULT = _env_float("GW_REVERB_DRY", 1.0)         # 干声比例 0.5~1.0
REVERB_ROOM_DEFAULT = _env_float("GW_REVERB_ROOM", 0.65)      # 房间大小 0~1（早反射分布）
REVERB_DECAY_DEFAULT = _env_float("GW_REVERB_DECAY", 1.4)     # 尾音长度（秒）0.2~4.0
REVERB_DAMPING_DEFAULT = _env_float("GW_REVERB_DAMPING", 0.45)  # 高频衰减 0~1
REVERB_PREDELAY_DEFAULT = _env_float("GW_REVERB_PREDELAY", 18.0)  # 预延迟（毫秒）0~100
REVERB_WIDTH_DEFAULT = _env_float("GW_REVERB_WIDTH", 0.80)    # 立体声宽度 0~1
REVERB_SR = 44100                    # 卷积统一工作采样率
REVERB_TAIL_MAX_SECONDS = _env_float("GW_REVERB_TAIL_MAX", 4.0)  # 输出最多比输入长这么多

# --- legacy aecho 回声（CUSTOM-6.9 的旧实现，默认关；与 reverb 互斥，reverb 优先） ---
ECHO_DELAYS_MS = "50|90|140"     # aecho 的 delays（毫秒），真实支持的参数
ECHO_WET_DEFAULT = 0.15
ECHO_DECAY_DEFAULT = 0.4         # aecho 的 decays（衰减系数，0~1）


def _as_bool(raw, default: bool = False) -> bool:
    """把 gradio/接口传入的开关值转成布尔；空值取默认，非法值报错。"""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("", "0", "false", "no", "off"):
        return False
    if text in ("1", "true", "yes", "on"):
        return True
    raise GatewayError(f"reverb_enabled 非法: {raw!r}（应为 true / false）")


def build_echo_filter(wet: float, dry: float, decay: float) -> tuple[str, dict]:
    """legacy aecho 回声滤镜（CUSTOM-6.9 旧实现，保留为高级可选效果）。

    返回 (滤镜串, 实际使用值)。固定 out_gain=1.0，湿声量完全由 decays 决定。
    """
    wet_v = float(wet)
    dry_v = float(dry)
    decay_v = max(0.05, min(0.9, float(decay)))
    scale = 1.0 + decay_v + (decay_v ** 2)
    d1 = wet_v / scale
    d2 = d1 * decay_v
    d3 = d2 * decay_v
    limiter = "alimiter=level_in=1:level_out=1:limit=0.95"
    if wet_v <= 0:
        af = f"volume={dry_v:.3f},{limiter}"
    else:
        af = (
            f"aecho=in_gain={dry_v:.3f}:out_gain=1.0"
            f":delays={ECHO_DELAYS_MS}:decays={d1:.3f}|{d2:.3f}|{d3:.3f}"
            f",{limiter}"
        )
    used = {
        "in_gain": round(dry_v, 3),
        "out_gain": 1.0,
        "delays_ms": ECHO_DELAYS_MS,
        "decays": f"{d1:.3f}|{d2:.3f}|{d3:.3f}",
        "decays_sum": round(d1 + d2 + d3, 3),
        "limiter": "alimiter(limit=0.95)",
    }
    return af, used


def apply_echo(src: Path, dest: Path, fmt: str, wet: float, dry: float,
               decay: float) -> tuple[float, dict]:
    """legacy aecho 后处理：写独立文件；返回 (耗时秒, 实际使用值)。

    失败时抛出 GatewayError，调用方保留原始 RVC 输出。
    """
    if not src.is_file():
        raise GatewayError(f"混响输入不存在: {src}")
    wet = _as_float(wet, ECHO_WET_DEFAULT, "echo_wet", 0.0, 0.5)
    dry = _as_float(dry, REVERB_DRY_DEFAULT, "echo_dry", 0.5, 1.0)
    decay = _as_float(decay, ECHO_DECAY_DEFAULT, "echo_decay", 0.05, 0.9)
    fmt = str(fmt or "wav").strip().lower()
    codec = {"wav": ["-c:a", "pcm_s16le"],
             "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
             "flac": ["-c:a", "flac"],
             "m4a": ["-c:a", "aac", "-b:a", "320k"]}.get(fmt)
    if not codec:
        raise GatewayError(f"混响不支持的导出格式: {fmt}")
    af, used = build_echo_filter(wet, dry, decay)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-i", str(src), "-af", af, *codec, str(dest)]
    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        try:
            dest.unlink(missing_ok=True)
        except OSError:
            pass
        raise GatewayError(f"混响处理失败({fmt}): {proc.stderr.strip()[-300:]}")
    os.chmod(dest, 0o644)
    return round(time.monotonic() - started, 3), used


# --------------------------------------------------------------------------- #
# 卷积混响（CUSTOM-7）：合成 IR + FFT 卷积，8 参数全部真实生效
# --------------------------------------------------------------------------- #

def _reverb_lowpass_noise(n: int, damping: float, rng) -> "np.ndarray":
    """生成一段已做高频衰减的白色噪声（IR 的弥散尾音）。

    damping=0 → 不过滤（明亮）；damping=1 → 单极点低通截止频率降到 ~500Hz（很暗）。
    用 scipy.signal.lfilter（C 实现）逐样本滤波，几万样本毫秒级完成。
    """
    import numpy as np
    from scipy.signal import lfilter

    noise = rng.standard_normal(n).astype(np.float32)
    damping_v = max(0.0, min(1.0, float(damping)))
    if damping_v <= 0.001:
        return noise
    # 截止频率从 12000Hz（无衰减）降到 500Hz（最暗），按 damping 幂次弯曲
    cutoff = 500.0 * (12000.0 / 500.0) ** (1.0 - damping_v)
    # 单极点低通：y[n] = y[n-1] + a * (x[n] - y[n-1])，a = 1 - exp(-2π·fc/fs)
    a = 1.0 - float(np.exp(-2.0 * np.pi * cutoff / REVERB_SR))
    a = min(max(a, 0.001), 0.999)
    return lfilter([a], [1.0, -(1.0 - a)], noise).astype(np.float32)


def build_reverb_ir(room_size: float, decay: float, damping: float,
                    pre_delay_ms: float, width: float) -> tuple["np.ndarray", "np.ndarray"]:
    """合成双声道脉冲响应（IR）。

    每个参数的真实作用（无假参数）：
      room_size  0~1  早反射的时间跨度与密度（大房间 → 早反射拉得更开）
      decay      秒   尾音总长度（指数衰减到 -60dB 的时间常数）
      damping    0~1  尾音的高频衰减速度（越大越暗）
      pre_delay  毫秒 IR 前面的静音（模拟声音到达第一面墙的延迟）
      width      0~1  左右尾音噪声的去相关程度（0=单声道，1=完全独立）

    返回 (ir_left, ir_right)，均为 float32、单位能量（sum(ir^2)=1），
    wet 参数因此可以直接理解为「湿声相对干声的能量比例」。
    """
    import numpy as np

    room_v = max(0.0, min(1.0, float(room_size)))
    decay_v = max(0.2, min(4.0, float(decay)))
    damping_v = max(0.0, min(1.0, float(damping)))
    pre_v = max(0.0, min(100.0, float(pre_delay_ms)))
    width_v = max(0.0, min(1.0, float(width)))

    rng = np.random.default_rng(20260130)  # 固定种子：IR 确定性，同参数同结果（利于缓存与复现）
    pre_n = int(round(pre_v / 1000.0 * REVERB_SR))
    tail_n = int(round(decay_v * REVERB_SR))

    # ---- 弥散尾音：指数衰减噪声，左右声道独立（按 width 混回单声道） ----
    tail_l = _reverb_lowpass_noise(tail_n, damping_v, rng)
    tail_r = _reverb_lowpass_noise(tail_n, damping_v, rng)
    mono = (tail_l + tail_r) / 2.0
    # width=0 → 双耳都是单声道信号；width=1 → 保持完全独立（去相关）
    tail_l = tail_l * width_v + mono * (1.0 - width_v)
    tail_r = tail_r * width_v + mono * (1.0 - width_v)
    t = np.arange(tail_n, dtype=np.float64) / REVERB_SR
    env = np.exp(-6.9078 * t / decay_v)  # -60dB(=20*log10(0.001)) 衰减
    tail_l = (tail_l * env).astype(np.float32)
    tail_r = (tail_r * env).astype(np.float32)

    # ---- 早反射：3~7 个稀疏脉冲，跨度随 room_size 拉开（大房间早反射更分明）----
    # 早反射两声道一致（居中）；立体声宽度只控制弥散尾音的去相关程度，
    # 这样 width=0 时左右 IR 完全相同（真正的单声道混响）。
    early_n = int(round((0.02 + 0.06 * room_v) * REVERB_SR))
    n_taps = 3 + int(round(4 * room_v))
    taps = np.zeros(max(early_n, 1), dtype=np.float32)
    for i in range(n_taps):
        pos = int((i + 1) / n_taps * max(early_n - 1, 1))
        taps[pos] += 0.7 ** i

    def assemble(taps: "np.ndarray", tail: "np.ndarray") -> "np.ndarray":
        ir = np.zeros(pre_n + max(early_n, 1) + tail_n, dtype=np.float32)
        ir[pre_n:pre_n + taps.size] += taps
        ir[pre_n + max(early_n // 2, 0): pre_n + max(early_n // 2, 0) + tail.size] += tail
        energy = float(np.sqrt(np.sum(np.square(ir, dtype=np.float64))))
        if energy > 1e-9:
            ir /= energy
        return ir

    return assemble(taps, tail_l), assemble(taps, tail_r)


def _decode_stereo_f32(src: Path) -> tuple["np.ndarray", int]:
    """把任意音频解码成 (N,2) float32（-1~1）。失败抛 GatewayError。"""
    import numpy as np

    cmd = ["ffmpeg", "-v", "error", "-i", str(src),
           "-ac", "2", "-ar", str(REVERB_SR), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not proc.stdout:
        raise GatewayError(f"混响解码失败: {proc.stderr.decode('utf-8', 'replace')[-200:]}")
    data = np.frombuffer(proc.stdout, dtype="<f4").reshape(-1, 2)
    if data.shape[0] == 0:
        raise GatewayError("混响解码结果为空")
    return data, REVERB_SR


def apply_reverb(src: Path, dest: Path, fmt: str, wet: float, dry: float,
                 room_size: float, decay: float, damping: float,
                 pre_delay: float, width: float) -> tuple[float, dict]:
    """卷积混响后处理：wet/dry 混合 + 峰值保护，写独立文件。

    输出保留自然尾音（长度 = 输入 + min(decay, TAIL_MAX)），不做硬截断。
    返回 (耗时秒, 实际使用值)。失败抛 GatewayError，调用方保留原始输出。
    """
    import numpy as np
    from scipy.signal import fftconvolve

    if not src.is_file():
        raise GatewayError(f"混响输入不存在: {src}")
    wet_v = _as_float(wet, REVERB_WET_DEFAULT, "reverb_wet", 0.0, 0.5)
    dry_v = _as_float(dry, REVERB_DRY_DEFAULT, "reverb_dry", 0.5, 1.0)
    room_v = _as_float(room_size, REVERB_ROOM_DEFAULT, "reverb_room_size", 0.0, 1.0)
    decay_v = _as_float(decay, REVERB_DECAY_DEFAULT, "reverb_decay", 0.2, 4.0)
    damping_v = _as_float(damping, REVERB_DAMPING_DEFAULT, "reverb_damping", 0.0, 1.0)
    pre_v = _as_float(pre_delay, REVERB_PREDELAY_DEFAULT, "reverb_pre_delay", 0.0, 100.0)
    width_v = _as_float(width, REVERB_WIDTH_DEFAULT, "reverb_width", 0.0, 1.0)

    fmt_l = str(fmt or "wav").strip().lower()
    if fmt_l not in ALLOWED_FORMATS:
        raise GatewayError(f"混响不支持的导出格式: {fmt_l}")

    started = time.monotonic()
    x, sr = _decode_stereo_f32(src)
    ir_l, ir_r = build_reverb_ir(room_v, decay_v, damping_v, pre_v, width_v)
    wet_l = fftconvolve(x[:, 0], ir_l).astype(np.float32)
    wet_r = fftconvolve(x[:, 1], ir_r).astype(np.float32)
    n_out = min(x.shape[0] + min(int(decay_v * sr), int(REVERB_TAIL_MAX_SECONDS * sr)),
                wet_l.size)
    # 干声补零到输出长度（保留自然尾音），避免两边长度不一致
    x_pad = np.zeros((n_out, 2), dtype=np.float32)
    copy_len = min(n_out, x.shape[0])
    x_pad[:copy_len] = x[:copy_len]
    mix = np.empty((n_out, 2), dtype=np.float32)
    mix[:, 0] = dry_v * x_pad[:, 0] + wet_v * wet_l[:n_out]
    mix[:, 1] = dry_v * x_pad[:, 1] + wet_v * wet_r[:n_out]
    # 峰值保护：仅在超限时按比例压回，不做压限染色
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    peak_scaled = False
    if peak > 0.95:
        mix *= 0.95 / peak
        peak_scaled = True

    # 经中间 wav 落盘（soundfile 直接写 float32），再转目标格式
    import soundfile as sf
    tmp_wav = dest.with_suffix(".revtmp.wav")
    try:
        sf.write(str(tmp_wav), mix, sr, subtype="PCM_16")
        convert_audio_fmt(tmp_wav, dest, fmt_l)
    finally:
        try:
            tmp_wav.unlink(missing_ok=True)
        except OSError:
            pass
    os.chmod(dest, 0o644)
    used = {
        "engine": "conv_ir",
        "wet": round(wet_v, 3), "dry": round(dry_v, 3),
        "room_size": round(room_v, 3), "decay_s": round(decay_v, 2),
        "damping": round(damping_v, 3), "pre_delay_ms": round(pre_v, 1),
        "width": round(width_v, 3),
        "ir_taps": int(ir_l.size), "peak_scaled": peak_scaled,
    }
    return round(time.monotonic() - started, 3), used


def ffmpeg_apply_fade(src: Path, fade_in: float, fade_out: float, dest: Path,
                      fmt: str, total: float) -> None:
    """给整段音频加首尾淡入淡出（秒），保持原格式编码。"""
    filters = []
    if fade_in > 0:
        filters.append(f"afade=t=in:st=0:d={fade_in:.3f}")
    if fade_out > 0:
        st = max(0.0, float(total) - fade_out)
        filters.append(f"afade=t=out:st={st:.3f}:d={fade_out:.3f}")
    if not filters:
        return
    codec = {"wav": ["-c:a", "pcm_s16le"],
             "mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
             "flac": ["-c:a", "flac"],
             "m4a": ["-c:a", "aac", "-b:a", "320k"]}.get(fmt, ["-c:a", "copy"])
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-af", ",".join(filters), *codec, str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError(f"淡入淡出处理失败: {proc.stderr.strip()[:400]}")


def ffmpeg_trim_fmt(src: Path, start: float, length: float, dest: Path, fmt: str) -> None:
    """按输出格式裁剪（wav 用 pcm，其他格式重编码，避免扩展名与编码不符）。"""
    if fmt == "wav":
        ffmpeg_trim(src, start, length, dest)
        return
    codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", "320k"],
             "flac": ["-c:a", "flac"],
             "m4a": ["-c:a", "aac", "-b:a", "320k"]}.get(fmt, ["-c:a", "copy"])
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
        "-map", "0:a:0", *codec, str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError(f"裁剪失败: {proc.stderr.strip()[:400]}")


def ffmpeg_trim(src: Path, start: float, length: float, dest: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-ss", f"{start:.3f}",
        "-t", f"{length:.3f}",
        "-map", "0:a:0",
        "-ac", "2", "-ar", "44100",
        "-c:a", "pcm_s16le",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError(f"FFmpeg 裁剪失败: {proc.stderr.strip()[:300]}")


def ffmpeg_apply_gain(src: Path, gain_db: float, dest: Path) -> None:
    """对成品施加音量增益（dB）。gain_db 为 0 时直接跳过。"""
    if abs(gain_db) < 0.01:
        return
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-af", f"volume={gain_db:.2f}dB",
        "-ar", "44100",
        str(dest),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        raise GatewayError(f"FFmpeg 音量调整失败: {proc.stderr.strip()[:300]}")


def load_envelope(path: Path):
    """把音频解码成单声道低采样率 PCM，返回 (samples_float32, sample_rate)。

    只在“挑副歌”这一步用，解码完即可释放；不会把整首歌送进分离/推理。
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(CHORUS_ANALYSIS_SR),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=FFMPEG_TIMEOUT)
    if proc.returncode != 0 or not proc.stdout:
        raise GatewayError("无法解码音频用于副歌分析")
    try:
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        raise GatewayError(f"缺少 numpy，无法做副歌分析: {exc}") from exc
    data = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0
    return data, CHORUS_ANALYSIS_SR


def audio_rms_db(src: Path) -> float:
    """算一段音频的平均电平（dBFS）；解码失败返回 -120（当作静音/不可用）。"""
    try:
        data, sr = load_envelope(src)
        if data.size == 0:
            return -120.0
        import numpy as np
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
        if rms <= 1e-8:
            return -120.0
        return 20.0 * float(np.log10(rms))
    except Exception as exc:  # noqa: BLE001
        logger.warning("电平检测失败 %s: %s", src, exc)
        return -120.0


def pick_chorus_window(total: float, target: float, src: Path, task_id: str,
                       min_target: float | None = None,
                       avoid_range: tuple[float, float] | None = None,
                       max_start: float | None = None):
    """轻量级副歌窗口挑选（初版，启发式）。

    做法：能量包络滑窗找平均能量最高的一段作为候选，再把首尾吸附到邻近的
    能量低谷（近似自然停顿/换气点）。

    ⚠️ 这不是可靠的副歌识别：它只找“整首里最响的一段”，可能是副歌，
    也可能是前奏高潮、间奏或桥段。需要准确副歌时请用 start_time/end_time 指定。

    任何一步失败都会安全降级为「从头取 target 秒」，绝不返回无效区间。

    ``min_target`` 给定时（CUSTOM-4.2）：窗口长度在 [min_target, target] 之间按 0.5 秒步进
    一起参与评分，取「得分最高」的那个长度 —— 也就是输出长度可以落在目标~目标+缓冲之间，
    而不是强行截成固定目标长度。
    """
    length = min(float(target), float(total))
    slack = max(total - length, 0.0)
    if total <= max(CHORUS_MIN_SECONDS, length + 0.5):
        # 音频太短，没什么可挑的；
        # 但变长模式下只要比 min_target 长，仍应进入多长度评分（否则永远返回整段）
        if min_target is None or total <= float(min_target) + 0.5:
            return 0.0, min(length, total), "too_short"

    def fallback(reason):
        logger.warning("[%s] 副歌挑选降级(%s)：改为从头取 %.2fs", task_id, reason, length)
        return 0.0, length, "fallback:" + reason

    try:
        data, sr = load_envelope(src)
        if data.size < sr:  # <1s，数据不可用
            return fallback("decoded_too_short")
        hop = max(1, int(CHORUS_ANALYSIS_SR * CHORUS_HOP_SECONDS))
        step = max(1, int(CHORUS_ANALYSIS_SR * CHORUS_STEP_SECONDS))
        win = max(1, int(length * sr))
        n_frames = data.size // hop
        if n_frames < 4:
            return fallback("too_few_frames")

        import numpy as np
        frames = np.abs(data[: n_frames * hop].reshape(n_frames, hop)).mean(axis=1)
        step_frames = max(1, step // hop)
        cumsum = np.concatenate(([0.0], np.cumsum(frames)))

        # 候选长度集合：固定模式只有 target；min_target 给定时在 [min_target, target] 间取若干档
        if min_target is not None and 0.5 <= float(min_target) < length - 0.05:
            span = length - float(min_target)
            steps = max(1, int(round(span / 0.5)))
            lengths = [float(min_target) + span * i / steps for i in range(steps + 1)]
        else:
            lengths = [length]
        var_mode = len(lengths) > 1

        best_score, best_start, best_len = -1.0, 0.0, length
        for cand_len in lengths:
            win_frames = max(1, int(cand_len * sr) // hop)
            if win_frames >= n_frames:
                continue
            starts = np.arange(0, n_frames - win_frames + 1, step_frames)
            if starts.size == 0:
                continue
            means = (cumsum[starts + win_frames] - cumsum[starts]) / win_frames
            window_stack = np.lib.stride_tricks.sliding_window_view(frames, win_frames)[starts]
            thresholds = np.percentile(window_stack, 60, axis=1)
            sustain = (window_stack >= thresholds[:, None]).mean(axis=1)
            del window_stack
            score = means * (0.7 + 0.3 * sustain)
            start_sec = starts * hop / float(sr)
            # 疑似前奏（开头 5s）与尾奏（结尾 5s）降权
            score = np.where(start_sec < 5.0, score * 0.85, score)
            score = np.where(start_sec + cand_len > total - 5.0, score * 0.9, score)
            if avoid_range is not None:
                # 避开上一轮判定为静音的区间（重叠的候选直接压到最低分）
                a0, a1 = avoid_range
                score = np.where(
                    (start_sec < a1) & (start_sec + cand_len > a0), -1.0, score)
            if max_start is not None:
                # 只在指定的起点上限内挑（重试时用来避开尾奏/outro）
                score = np.where(start_sec > max_start, -1.0, score)
            # 轻微长度偏好：除非明显更好，否则不轻易把窗口拉长
            if length > float(min_target or length) + 0.05:
                span = max(length - float(min_target), 0.5)
                score = score * (1.0 - LENGTH_PREFERENCE * (cand_len - float(min_target)) / span)
            i = int(np.argmax(score))
            if float(score[i]) > best_score:
                best_score, best_start, best_len = float(score[i]), float(start_sec[i]), float(cand_len)

        if best_score < 0:
            return fallback("no_candidate_window")
        length = best_len
        slack = max(total - length, 0.0)   # 长度变了，可滑动余量要重算
        win_frames = max(1, int(length * sr) // hop)
        # 复算该长度下的评分数组，供对比度统计使用
        starts = np.arange(0, n_frames - win_frames + 1, step_frames)
        means = (cumsum[starts + win_frames] - cumsum[starts]) / win_frames
        window_stack = np.lib.stride_tricks.sliding_window_view(frames, win_frames)[starts]
        thresholds = np.percentile(window_stack, 60, axis=1)
        sustain = (window_stack >= thresholds[:, None]).mean(axis=1)
        del window_stack
        score = means * (0.7 + 0.3 * sustain)
        start_sec = starts * hop / float(sr)
        score = np.where(start_sec < 5.0, score * 0.85, score)
        score = np.where(start_sec + length > total - 5.0, score * 0.9, score)
        best_i = int(np.argmin(np.abs(start_sec - best_start)))
        # 可分辨度：最优分与中位分的相对差太小 => 各段差别不大，谈不上“挑出副歌”
        contrast = float((score[best_i] - np.median(score)) / max(float(score[best_i]), 1e-9))
        low_contrast = contrast < 0.05

        # ---- 首尾双向吸附到邻近能量低谷（近似停顿/换气点）----
        snap_frames = max(1, int(CHORUS_SNAP_WINDOW * sr / hop))
        s_idx = best_i * step_frames

        def snap_to_min(center_idx, radius):
            lo = max(0, center_idx - radius)
            hi = min(n_frames - 1, center_idx + radius)
            return int(lo + int(np.argmin(frames[lo:hi + 1])))

        # 吸附规则：只挪动窗口位置，**保持请求长度不变**（用户要 20 秒就给 20 秒）
        snapped_s = snap_to_min(s_idx, snap_frames)
        start = float(snapped_s * hop) / sr
        end = start + length
        # 再尝试把终点也吸到低谷：终点吸附后把起点整体回移，长度仍为 length
        e_idx = min(int(round((start + length) * sr / hop)), n_frames - 1)
        snapped_e = snap_to_min(e_idx, snap_frames)
        end2 = float(snapped_e * hop) / sr
        start2 = end2 - length
        if start2 >= 0 and start2 <= slack:
            start, end = start2, end2
        new_len = end - start

        # ---- 边界检查（绝不允许越界 / 非正数 / 超过 30 秒上限）----
        start = max(0.0, min(start, slack))
        new_len = max(0.5, min(new_len, length, total - start, MAX_CLIP_SECONDS))
        if start + new_len > total:
            new_len = max(0.5, total - start)

        tag = "chorus_picked_var" if var_mode else "chorus_picked"
        if low_contrast:
            tag = "chorus_picked_low_contrast"
        logger.info(
            "[%s] 副歌挑选(启发式,非可靠识别)：%.2fs~%.2fs | 候选=%d 评分对比度=%.3f%s | "
            "起点吸附 %+.2fs 终点吸附 %+.2fs",
            task_id, start, start + new_len, len(starts), contrast,
            "（低，各段差异不明显，仅供参考）" if low_contrast else "",
            start - best_start, (start + new_len) - (best_start + length))
        return round(start, 3), round(new_len, 3), tag
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] 副歌挑选异常(%s)，安全降级", task_id, exc)
        return fallback("exception")


# --------------------------------------------------------------------------- #
# 时间参数解析与校验
# --------------------------------------------------------------------------- #

_TIME_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def parse_time_value(raw, name: str) -> float | None:
    """支持 '80' / '80s' / '01:20' / '1:20.5'；空值返回 None。"""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("秒", "").replace("s", "").replace("S", "").strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) > 3 or not all(p.strip() for p in parts):
            raise GatewayError(f"{name} 格式非法: {raw!r}")
        try:
            total = 0.0
            for part in parts:
                total = total * 60 + float(part)
        except ValueError as exc:
            raise GatewayError(f"{name} 格式非法: {raw!r}") from exc
        return total
    if not _TIME_RE.match(text):
        raise GatewayError(f"{name} 格式非法: {raw!r}")
    value = float(text)
    if value < 0:
        raise GatewayError(f"{name} 不能为负数: {raw!r}")
    return value


def compute_clip_target(requested, total, buffer_seconds=None, max_clip=None) -> float:
    """chorus 候选切片长度 = 目标时长 + 缓冲，受源音频总长与硬上限约束。

    不越界、不伪造：源音频不够就只取实际长度；目标为 0/非法时按 0 处理。
    """
    buffer_seconds = SEP_BUFFER_SECONDS if buffer_seconds is None else buffer_seconds
    max_clip = MAX_CLIP_SECONDS if max_clip is None else max_clip
    try:
        want = float(requested) + max(0.0, float(buffer_seconds))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(want, float(max_clip), float(total)))


def resolve_clip(total: float, start_time, end_time, duration):
    """返回 (start, length)，违反规则一律 GatewayError（不做静默截断）。"""
    start = parse_time_value(start_time, "start_time")
    end = parse_time_value(end_time, "end_time")
    dur = parse_time_value(duration, "duration")

    if start is not None and start >= total:
        raise GatewayError(
            f"开始时间 {start:.2f}s 已超出歌曲总时长 {total:.2f}s，"
            f"可用范围 0 ~ {max(total - 0.01, 0):.2f}s。"
        )
    if end is not None and end > total:
        raise GatewayError(
            f"结束时间 {end:.2f}s 超过歌曲总时长 {total:.2f}s，"
            f"可用范围 0 ~ {total:.2f}s。"
        )
    if start is not None and end is not None:
        if end <= start:
            raise GatewayError(f"结束时间 {end:.2f}s 必须大于开始时间 {start:.2f}s。")
        length = end - start
        if length > MAX_CLIP_SECONDS:
            raise GatewayError(
                f"片段长度 {length:.2f}s 超过上限 {MAX_CLIP_SECONDS:.0f}s，请缩短区间。"
            )
        return start, length

    if start is not None and dur is not None:
        length = dur
    elif start is not None:
        length = DEFAULT_CLIP_SECONDS
    elif dur is not None:
        start = 0.0
        length = dur
    else:
        raise GatewayError(
            "未提供片段起止时间，且网关没有可靠的高潮/副歌时间信息，"
            "无法定位片段。请提供 start_time 或 duration（例如 start_time='01:20'）。"
        )

    if length <= 0:
        raise GatewayError("片段长度必须大于 0。")
    if length > MAX_CLIP_SECONDS:
        raise GatewayError(
            f"片段长度 {length:.2f}s 超过上限 {MAX_CLIP_SECONDS:.0f}s。"
        )
    if start + length > total:
        raise GatewayError(
            f"片段 {start:.2f}s ~ {start + length:.2f}s 超出歌曲总时长 {total:.2f}s，"
            f"从 {start:.2f}s 起最多可取 {max(total - start, 0):.2f}s。"
        )
    return start, length


# --------------------------------------------------------------------------- #
# 推理参数校验
# --------------------------------------------------------------------------- #

def _as_float(raw, default: float, name: str, lo: float, hi: float) -> float:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise GatewayError(f"{name} 非法: {raw!r}") from exc
    if not (lo <= value <= hi):
        raise GatewayError(f"{name} 超出允许范围 [{lo}, {hi}]: {value}")
    return value


def _as_int(raw, default: int, name: str, lo: int, hi: int) -> int:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = int(float(raw))
    except (TypeError, ValueError) as exc:
        raise GatewayError(f"{name} 非法: {raw!r}") from exc
    if not (lo <= value <= hi):
        raise GatewayError(f"{name} 超出允许范围 [{lo}, {hi}]: {value}")
    return value


def _as_choice(raw, default: str, name: str, choices) -> str:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    value = str(raw).strip()
    if value not in choices:
        raise GatewayError(f"{name} 不支持: {value!r}（可用: {', '.join(choices)}）")
    return value


# --------------------------------------------------------------------------- #
# 子进程执行（统一超时 + 登记，便于取消）
# --------------------------------------------------------------------------- #

def run_stage(task_id: str, cmd: list[str], cwd: str, env: dict, timeout: int, stage: str,
              progress=None, pct=0.0):
    """启动子进程并**周期性 yield**，使 Gradio 取消时的 GeneratorExit 能被送达。

    这是取消功能的关键：如果一直阻塞在 communicate()，生成器无法被关闭，
    job.cancel() 就杀不掉底层进程。轮询 + yield 可以让取消真正生效。
    """
    logger.info("[%s] %s 启动: %s", task_id, stage, " ".join(cmd))
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    _register_proc(task_id, proc)
    while proc.poll() is None:
        if time.monotonic() - started > timeout:
            _kill_procs(task_id)
            raise GatewayError(f"{stage} 超时（>{timeout}s），已终止子进程")
        if progress is not None:
            progress(pct, desc=f"{stage}中")
        yield f"{stage}中（{time.monotonic() - started:.0f}s）"
        time.sleep(0.5)
    out, _ = proc.communicate()
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = (out or "").strip()[-600:]
        if proc.returncode == -9:
            raise GatewayError(
                f"{stage} 被系统终止（退出码 -9，通常是内存不足 OOM）。"
                f"请缩短片段时长或提高容器内存上限。"
            )
        raise GatewayError(f"{stage} 失败（退出码 {proc.returncode}）：{tail}")
    logger.info("[%s] %s 完成，耗时 %.2fs", task_id, stage, elapsed)
    return out or "", elapsed


# --------------------------------------------------------------------------- #
# 缓存（仅管理 /work/cache，绝不动模型、音源、其他任务文件）
# --------------------------------------------------------------------------- #

CACHE_DIR = Path(CONTAINER_WORK_DIR) / "cache"
OUT_DIR = Path(CONTAINER_WORK_DIR) / "out"
TASK_ROOT = Path(CONTAINER_WORK_DIR) / "tasks"
LOG_DIR = Path(CONTAINER_WORK_DIR) / "logs"
# CUSTOM-7：TTS 语音转换的输入目录（AstrBot 宿主机直接写入 gateway_work/tts/<task_id>/）
TTS_ROOT = Path(CONTAINER_WORK_DIR) / "tts"
TTS_MAX_SECONDS = _env_float("GW_TTS_MAX_SECONDS", 60.0)
TTS_MAX_BYTES = _env_int("GW_TTS_MAX_BYTES", 30 * 1024 * 1024)
TTS_TTL_SECONDS = _env_int("GW_TTS_TTL", 3600)


def _ensure_dirs() -> None:
    for d in (CACHE_DIR, OUT_DIR, TASK_ROOT, LOG_DIR, TTS_ROOT):
        d.mkdir(parents=True, exist_ok=True)
        try:
            # TTS_ROOT 特殊：宿主机 AstrBot 进程（非容器用户）也要往里写原始音频，
            # 因此放开为全员可写（仅这一个目录，非递归）
            os.chmod(d, 0o777 if d == TTS_ROOT else 0o755)
        except OSError as exc:  # pragma: no cover
            if exc.errno != errno.EPERM:
                raise


def _cache_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _cache_read(key: str) -> dict | None:
    meta = CACHE_DIR / f"{key}.json"
    if not meta.is_file():
        return None
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    path = Path(data.get("output", ""))
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return data


def cache_info() -> dict:
    """插件契约：返回 dict，含 total_files / total_bytes / areas。"""
    _ensure_dirs()
    areas: dict[str, dict] = {}
    total_files = 0
    total_bytes = 0
    for area, path in (("cache", CACHE_DIR), ("out", OUT_DIR), ("tasks", TASK_ROOT), ("logs", LOG_DIR)):
        files = 0
        size = 0
        if path.is_dir():
            for p in path.rglob("*"):
                if p.is_file():
                    files += 1
                    try:
                        size += p.stat().st_size
                    except OSError:
                        pass
        areas[area] = {"files": files, "bytes": size}
        total_files += files
        total_bytes += size
    return {"total_files": total_files, "total_bytes": total_bytes, "areas": areas}


def clear_cache(scope: str = "all") -> dict:
    """只清理网关自己的结果缓存目录 /work/cache；不动模型、音源与其他任务产物。"""
    _ensure_dirs()
    deleted = 0
    freed = 0
    scope = str(scope or "all").lower()
    targets = []
    if scope in ("all", "cache", "results"):
        targets.append(CACHE_DIR)
    if scope in ("all", "out"):
        targets.append(OUT_DIR)
    for base in targets:
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    deleted += 1
                except OSError:
                    pass
    logger.info("clear_cache scope=%s deleted=%d freed=%d", scope, deleted, freed)
    return {"deleted_files": deleted, "freed_bytes": freed, "scope": scope}


# --------------------------------------------------------------------------- #
# 主转换流程
# --------------------------------------------------------------------------- #

def _sep_stage(task_id: str, clip: Path, sep_dir: Path, sep_model: str, progress=None):
    """生成器：yield 进度，最终 return (vocals_path, elapsed)。"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(APP_DIR, "tools")
    env["PYMSS_MODEL_DIR"] = PYMSS_MODEL_DIR
    env["OMP_NUM_THREADS"] = OMP_THREADS
    sep_model_path(sep_model)
    cmd = [
        SYSTEM_PYTHON, "-m", "pymss.cli", "infer", sep_model,
        "-i", str(clip), "-o", str(sep_dir),
        "--device", "cpu", "--format", "wav",
        "--param", "batch_size=1",
    ]
    _, elapsed = yield from run_stage(
        task_id, cmd, cwd=APP_DIR, env=env, timeout=SEP_TIMEOUT,
        stage="人声分离", progress=progress, pct=0.45)
    vocals = sorted(sep_dir.rglob("*_Vocals.wav"))
    if not vocals:
        raise GatewayError(f"人声分离未产出 Vocals 文件（目录: {sep_dir}）")
    return vocals[0], elapsed


def _rvc_stage(task_id: str, vocals: Path, out_file: Path, params: dict, progress=None):
    """生成器：yield 进度，最终 return (elapsed, index_path)。【子进程回退路径】"""
    env = dict(os.environ)
    # 直接以脚本方式运行 /app/infer/cli.py 时 sys.path[0] 是 /app/infer，
    # 必须显式把 /app 加进 PYTHONPATH，否则 cli.py 里的 `from infer...` 会失败。
    env["PYTHONPATH"] = APP_DIR
    env.setdefault("weight_root", os.path.join(APP_DIR, "assets", "weights"))
    env.setdefault("outside_index_root", os.path.join(APP_DIR, "assets", "indices"))
    env.setdefault("rmvpe_root", os.path.join(APP_DIR, "assets", "rmvpe"))
    env["OMP_NUM_THREADS"] = OMP_THREADS

    index_path = params["index_path"]
    cmd = [
        SYSTEM_PYTHON, os.path.join(APP_DIR, "infer", "cli.py"),
        "--model", params["model"],
        "--input", str(vocals),
        "--output", str(out_file),
        "--pitch", str(params["pitch"]),
        "--f0-method", params["f0_method"],
        "--index-rate", f"{params['index_rate']:.4f}",
        "--protect", f"{params['protect']:.4f}",
        "--resample-sr", str(params["resample_sr"]),
        "--rms-mix-rate", f"{params['rms_mix_rate']:.4f}",
        "--format", params["output_format"],
        "--overwrite",
    ]
    if index_path:
        cmd += ["--index", index_path]
    if params.get("speaker_id"):
        cmd += ["--speaker-id", str(params["speaker_id"])]
    _, elapsed = yield from run_stage(
        task_id, cmd, cwd=APP_DIR, env=env, timeout=RVC_TIMEOUT,
        stage="RVC推理", progress=progress, pct=0.8)
    return elapsed, index_path


# --------------------------------------------------------------------------- #
# 常驻 RVC worker（CUSTOM-7.2）
# --------------------------------------------------------------------------- #

class RvcWorker:
    """常驻 RVC 推理进程：保持模型在内存，省掉每次 ~40s 的模型加载。

    协议：JSON Lines over stdio（协议通道是 worker 复制的 stdout）。
    任何失败（崩溃/超时/OOM）都抛 GatewayError，由调用方回退子进程路径。
    空闲 WORKER_IDLE_SECONDS 秒后 worker 自行退出，下次请求自动重新拉起。
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _spawn(self) -> None:
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_worker.py")
        env = dict(os.environ)
        env["PYTHONPATH"] = APP_DIR
        env.setdefault("weight_root", os.path.join(APP_DIR, "assets", "weights"))
        env.setdefault("outside_index_root", os.path.join(APP_DIR, "assets", "indices"))
        env.setdefault("rmvpe_root", os.path.join(APP_DIR, "assets", "rmvpe"))
        env["OMP_NUM_THREADS"] = OMP_THREADS
        env["GW_WORKER_IDLE_SECONDS"] = str(WORKER_IDLE_SECONDS)
        self._proc = subprocess.Popen(
            [SYSTEM_PYTHON, script],
            cwd=APP_DIR, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._drain, args=(self._proc.stdout, self._lines), daemon=True).start()
        threading.Thread(target=self._drain_err, args=(self._proc.stderr,), daemon=True).start()
        logger.info("[worker] 已启动（pid=%s，空闲 %ds 自动退出）", self._proc.pid, WORKER_IDLE_SECONDS)

    def _drain(self, stream, lines: queue.Queue) -> None:
        try:
            for line in stream:
                lines.put(line.rstrip("\n"))
        except Exception:
            pass
        lines.put(None)

    def _drain_err(self, stream) -> None:
        try:
            for line in stream:
                text = line.rstrip("\n")
                if text.strip():
                    logger.debug("[worker] %s", text)
        except Exception:
            pass

    def infer(self, req: dict, timeout: int) -> dict:
        with self._lock:
            if not self.alive():
                self.kill()
                self._spawn()
            rid = uuid.uuid4().hex[:8]
            payload = dict(req)
            payload["id"] = rid
            try:
                self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self._proc.stdin.flush()
            except Exception as exc:
                self.kill()
                raise GatewayError(f"worker 通信失败: {exc}") from exc
            deadline = time.monotonic() + max(10, int(timeout))
            while True:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    self.kill()
                    raise GatewayError(f"worker 推理超时（>{timeout}s），已终止 worker")
                try:
                    line = self._lines.get(timeout=min(remain, 1.0))
                except queue.Empty:
                    continue
                if line is None:
                    self.kill()
                    raise GatewayError("worker 进程意外退出（可能内存不足被 OOM kill）")
                try:
                    resp = json.loads(line)
                except Exception:
                    logger.debug("[worker] 非协议输出: %s", str(line)[:200])
                    continue
                if resp.get("event") in ("ready", "idle_exit"):
                    continue
                if resp.get("id") != rid:
                    continue
                if not resp.get("ok"):
                    raise GatewayError(f"worker 推理失败: {resp.get('error')}")
                return resp

    def kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            except Exception:
                pass
            logger.info("[worker] 已终止")


_WORKER = RvcWorker()
atexit.register(_WORKER.kill)


def run_rvc_stage(task_id: str, vocals: Path, out_file: Path, params: dict, progress=None):
    """RVC 推理统一入口：优先常驻 worker（热路径省 ~40s 模型加载），失败回退子进程。"""
    if not WORKER_ENABLED:
        return (yield from _rvc_stage(task_id, vocals, out_file, params, progress))
    try:
        elapsed = yield from _worker_stage(task_id, vocals, out_file, params, progress)
        return elapsed, params.get("index_path") or ""
    except Exception as exc:  # noqa: BLE001 — worker 任何问题都回退到经过验证的子进程路径
        logger.warning("[%s] worker 推理失败（%s），回退子进程推理", task_id, exc)
        _WORKER.kill()
        return (yield from _rvc_stage(task_id, vocals, out_file, params, progress))


def _worker_stage(task_id: str, vocals: Path, out_file: Path, params: dict, progress=None):
    """生成器：yield 进度，最终 return worker 上报的 elapsed 秒。"""
    started = time.monotonic()
    if progress is not None:
        progress(0.75, desc="RVC 推理（常驻模型）")
    yield "RVC 推理（常驻模型）"
    resp = _WORKER.infer({
        "model": params["model"],
        "input": str(vocals),
        "output": str(out_file),
        "pitch": params["pitch"],
        "f0_method": params["f0_method"],
        "index": params.get("index_path") or "",
        "index_rate": params["index_rate"],
        "resample_sr": params.get("resample_sr", 0),
        "rms_mix_rate": params["rms_mix_rate"],
        "protect": params["protect"],
        "output_format": params["output_format"],
        "speaker_id": params.get("speaker_id", 0),
    }, timeout=RVC_TIMEOUT)
    elapsed = float(resp.get("elapsed") or (time.monotonic() - started))
    logger.info("[%s] worker 推理完成，耗时 %.2fs（模型 %s）", task_id, elapsed, params["model"])
    return elapsed


def convert(
    song_name_src,
    key_shift,
    vocal_vol,
    inst_vol,
    model_dropdown,
    reverb_intensity,
    delay_intensity,
    uvr5_agg,
    uvr5_tta,
    uvr5_postprocess,
    uvr5_window_size,
    uvr5_high_end_process,
    msst_batch_size,
    msst_num_overlap,
    msst_normalize,
    vocal_postprocess,
    shift_accompaniment,
    f0_method,
    index_rate,
    filter_radius,
    msst_model="",
    msst_use_tta=False,
    start_time="",
    end_time="",
    duration="",
    output_format="",
    speaker_id=0,
    # ---- chorus-mw 契约扩展（插件只有探测到这些名字才会下发）----
    render_mode="",
    chorus_seconds=0,
    rms_mix_rate=None,
    protect=None,
    inst_volume_db=None,
    vocal_volume_db=None,
    # CUSTOM-5：首尾淡入淡出时长（秒），0=不处理；插件按开关传 0.5
    fade_seconds=0.0,
    reverb_enabled=False,
    reverb_wet=REVERB_WET_DEFAULT,
    reverb_dry=REVERB_DRY_DEFAULT,
    reverb_decay=REVERB_DECAY_DEFAULT,
    # CUSTOM-7：卷积混响扩展参数（旧调用方不传时用默认值）
    reverb_room_size=REVERB_ROOM_DEFAULT,
    reverb_damping=REVERB_DAMPING_DEFAULT,
    reverb_pre_delay=REVERB_PREDELAY_DEFAULT,
    reverb_width=REVERB_WIDTH_DEFAULT,
    # CUSTOM-7：legacy aecho 回声（默认关；reverb 开启时忽略）
    echo_enabled=False,
    echo_wet=ECHO_WET_DEFAULT,
    echo_decay=ECHO_DECAY_DEFAULT,
    progress=gr.Progress(),
):
    """插件 /convert 的契约实现。返回宿主机可读取的结果文件绝对路径。

    片段来源优先级：
      1. 显式 start_time / end_time / duration  -> 严格校验，越界即拒绝
      2. 否则按 render_mode + chorus_seconds    -> 启发式挑窗口（最长 30s）
      3. 再不行 -> 安全降级：从头取 min(chorus_seconds, 30, 总时长)
    """
    task_id = uuid.uuid4().hex[:12]
    task_dir = TASK_ROOT / task_id
    acquired = False
    try:
        _ensure_dirs()
        sweep_stale_tasks()
        # 全局串行：其他任务（歌曲/TTS）未完成时**等待**而非拒绝，
        # 等待期间周期性 yield 进度（取消时 GeneratorExit 正常退出，不会占用槽位）
        waited = 0.0
        while not _TASK_SLOT.acquire(blocking=False):
            waited += 0.5
            if waited > _env_float("GW_SLOT_WAIT", 180.0):
                raise GatewayError("已有推理任务在执行且等待超时，请稍后再试（网关全局串行）。")
            progress(0.02, desc="等待其他任务完成")
            yield f"等待其他任务完成（{waited:.0f}s）"
            time.sleep(0.5)
        acquired = True

        task_dir.mkdir(parents=True, exist_ok=True)
        sep_dir = task_dir / "sep"
        sep_dir.mkdir(parents=True, exist_ok=True)

        # ---------- 1. 参数校验 ----------
        available = list_rvc_models()
        if not available:
            raise GatewayError(f"音色目录为空: {WEIGHTS_DIR}")
        model = str(model_dropdown or "").strip()
        if not model:
            model = available[0]
        if model not in available:
            raise GatewayError(f"音色模型不存在: {model}（可用: {', '.join(available)}）")

        # chorus-mw 语义：分离模型由后端配置决定，插件传来的只是提示。
        # 因此请求的模型不可用时**不报错**，改为回退到网关配置的默认模型，
        # 并在日志与任务记录里同时保留「请求值」与「实际值」，绝不静默。
        requested_sep = str(msst_model or "").strip() or SEP_MODEL_DEFAULT
        sep_model = requested_sep
        try:
            sep_model_path(sep_model)
        except GatewayError as exc:
            logger.warning(
                "[%s] 请求的分离模型 %s 不可用（%s），回退到网关默认 %s",
                task_id, requested_sep, exc, SEP_MODEL_DEFAULT)
            sep_model = SEP_MODEL_DEFAULT
            sep_model_path(sep_model)

        pitch = _as_int(key_shift, DEFAULT_PITCH, "key_shift", -24, 24)
        f0 = _as_choice(f0_method, DEFAULT_F0_METHOD, "f0_method", ALLOWED_F0)
        idx_rate = _as_float(index_rate, DEFAULT_INDEX_RATE, "index_rate", 0.0, 1.0)
        protect = _as_float(protect, DEFAULT_PROTECT, "protect", 0.0, 0.5)
        resample_sr = _as_int(None, DEFAULT_RESAMPLE_SR, "resample_sr", 0, 48000)
        rms_mix = _as_float(rms_mix_rate, DEFAULT_RMS_MIX_RATE, "rms_mix_rate", 0.0, 1.0)
        fmt = _as_choice(output_format, DEFAULT_OUTPUT_FORMAT, "output_format", ALLOWED_FORMATS)
        sid = _as_int(speaker_id, 0, "speaker_id", 0, 4096)
        vocal_gain = _as_float(vocal_volume_db, 0.0, "vocal_volume_db", -12.0, 12.0)
        fade = _as_float(fade_seconds, 0.0, "fade_seconds", 0.0, 5.0)
        reverb_on = _as_bool(reverb_enabled, False)
        rev_wet = _as_float(reverb_wet, REVERB_WET_DEFAULT, "reverb_wet", 0.0, 0.5)
        rev_dry = _as_float(reverb_dry, REVERB_DRY_DEFAULT, "reverb_dry", 0.5, 1.0)
        rev_decay = _as_float(reverb_decay, REVERB_DECAY_DEFAULT, "reverb_decay", 0.2, 4.0)
        rev_room = _as_float(reverb_room_size, REVERB_ROOM_DEFAULT, "reverb_room_size", 0.0, 1.0)
        rev_damp = _as_float(reverb_damping, REVERB_DAMPING_DEFAULT, "reverb_damping", 0.0, 1.0)
        rev_pre = _as_float(reverb_pre_delay, REVERB_PREDELAY_DEFAULT, "reverb_pre_delay", 0.0, 100.0)
        rev_width = _as_float(reverb_width, REVERB_WIDTH_DEFAULT, "reverb_width", 0.0, 1.0)
        echo_on = _as_bool(echo_enabled, False)
        echo_wet_v = _as_float(echo_wet, ECHO_WET_DEFAULT, "echo_wet", 0.0, 0.5)
        echo_decay_v = _as_float(echo_decay, ECHO_DECAY_DEFAULT, "echo_decay", 0.05, 0.9)

        # render_mode：full 会要求先处理整首歌，与「先裁剪再分离/推理」冲突，明确拒绝
        rmode = _as_choice(render_mode or "chorus", "chorus", "render_mode",
                           ("chorus", "custom", "full"))
        if rmode == "full":
            raise GatewayError(
                "render_mode=full 需要对整首歌做分离与推理，本机内存与 CPU 不允许；"
                f"请改用 chorus 或 custom（最长 {MAX_CLIP_SECONDS:.0f} 秒）。")
        # 目标时长：请求值再大也最多 MAX_CLIP_SECONDS
        requested = _as_float(chorus_seconds, DEFAULT_CLIP_SECONDS, "chorus_seconds",
                              1.0, MAX_INPUT_SECONDS)
        target = min(requested, MAX_CLIP_SECONDS)
        if requested > MAX_CLIP_SECONDS:
            logger.warning("[%s] chorus_seconds=%.1f 超过上限，按 %.1fs 裁剪",
                           task_id, requested, MAX_CLIP_SECONDS)
        if requested + SEP_BUFFER_SECONDS > 60:
            logger.warning(
                "[%s] 目标 %.1fs + 缓冲 %.1fs 超过 60 秒，8GB NAS 上分离/推理会很慢甚至 OOM；"
                "建议 60 秒以内", task_id, requested, SEP_BUFFER_SECONDS)

        # 插件传入但本网关/CLI 不支持的参数：明确记录为“已忽略”，绝不假装生效
        unsupported = {
            "vocal_vol": vocal_vol, "inst_vol": inst_vol,
            "reverb_intensity": reverb_intensity, "delay_intensity": delay_intensity,
            "uvr5_agg": uvr5_agg, "uvr5_tta": uvr5_tta,
            "uvr5_postprocess": uvr5_postprocess, "uvr5_window_size": uvr5_window_size,
            "uvr5_high_end_process": uvr5_high_end_process,
            "msst_num_overlap": msst_num_overlap, "msst_normalize": msst_normalize,
            "msst_use_tta": msst_use_tta, "vocal_postprocess": vocal_postprocess,
            "shift_accompaniment": shift_accompaniment, "filter_radius": filter_radius,
            # 本网关只出人声、不混伴奏，因此伴奏电平无从应用
            "inst_volume_db": inst_volume_db,
        }
        logger.info("[%s] 已接收但当前未生效的参数(记录备查): %s", task_id, unsupported)

        index_path = find_index_for(model)
        if index_path:
            logger.info("[%s] 音色 %s 使用 index: %s", task_id, model, index_path)
            effective_index_rate = idx_rate
        else:
            # infer/cli.py 在无 index 且 index_rate>0 时会直接报错，
            # 因此无索引模型强制走 index_rate=0 的无索引推理路径。
            effective_index_rate = 0.0
            logger.warning(
                "[%s] 音色 %s 未找到配对 index，改为无索引推理（index_rate %.2f -> 0.0）",
                task_id, model, idx_rate)

        # ---------- 2. 取音源 ----------
        progress(0.05, desc="获取音源")
        src = resolve_source(song_name_src, task_dir)
        size = src.stat().st_size
        if size > MAX_INPUT_BYTES:
            raise GatewayError(f"音源文件过大: {size / 1048576:.1f}MB（上限 {MAX_INPUT_BYTES / 1048576:.0f}MB）")

        # ---------- 3. 时长 & 裁剪校验 ----------
        total = probe_duration(src)
        if total > MAX_INPUT_SECONDS:
            raise GatewayError(f"音源过长: {total:.1f}s（上限 {MAX_INPUT_SECONDS:.0f}s）")
        has_explicit = any(
            parse_time_value(v, n) is not None
            for v, n in ((start_time, "start_time"), (end_time, "end_time"), (duration, "duration"))
        )
        if has_explicit:
            # 用户/调用方显式给了区间：严格校验，越界一律拒绝（不静默截断）
            start, length = resolve_clip(total, start_time, end_time, duration)
            clip_source = "explicit"
        elif rmode == "custom":
            # custom：按请求长度从头取（不做副歌挑选）
            length = min(target, total)
            start = 0.0
            clip_source = "custom_head"
        else:
            # chorus：候选切片 = 目标 + 缓冲（CUSTOM-4：15s -> 5s），挑窗口 + 停顿吸附
            target = compute_clip_target(requested, total)
            start, length, clip_source = pick_chorus_window(total, target, src, task_id)

        # 兜底：无论如何都不允许越界 / 非正数 / 超过 30 秒
        if start < 0 or start >= total or length <= 0:
            raise GatewayError(f"裁剪区间非法: start={start:.3f} length={length:.3f} total={total:.3f}")
        length = min(length, MAX_CLIP_SECONDS, total - start)
        logger.info(
            "[%s] 音源时长 %.2fs，候选片段 %.2fs ~ %.2fs（来源=%s，目标=%.1fs，缓冲=%.1fs，候选=%.1fs，上限=%.0fs）",
            task_id, total, start, start + length, clip_source,
            requested, SEP_BUFFER_SECONDS, target, MAX_CLIP_SECONDS)

        # ---------- 4. 缓存 ----------
        cache_payload = {
            "src": str(src), "model": model, "sep_model": sep_model,
            "start": round(start, 3), "length": round(length, 3),
            "pitch": pitch, "f0": f0, "index_rate": idx_rate, "protect": protect,
            "resample_sr": resample_sr, "rms_mix_rate": rms_mix, "format": fmt,
            "speaker_id": sid, "index": index_path or "",
            "vocal_volume_db": vocal_gain, "render_mode": rmode,
            "chorus_seconds_requested": round(requested, 3), "clip_source": clip_source,
            "fade_seconds": round(fade, 2),
            "reverb": [reverb_on, round(rev_wet, 3), round(rev_dry, 3), round(rev_decay, 3),
                       round(rev_room, 3), round(rev_damp, 3), round(rev_pre, 1), round(rev_width, 3)],
            "echo": [echo_on, round(echo_wet_v, 3), round(echo_decay_v, 3)],
        }
        key = _cache_key(cache_payload)
        hit = _cache_read(key)
        if hit:
            progress(1.0, desc="缓存命中")
            logger.info("[%s] 缓存命中 -> %s", task_id, hit.get("output"))
            yield to_host_path(hit["output"])
            return

        # ---------- 5. 裁剪 ----------
        progress(0.15, desc="裁剪片段")
        clip = task_dir / "clip.wav"
        t0 = time.monotonic()
        ffmpeg_trim(src, start, length, clip)
        trim_elapsed = time.monotonic() - t0
        clip_dur = probe_duration(clip)
        logger.info("[%s] 裁剪完成 %.2fs（实际 %.2fs），耗时 %.2fs",
                    task_id, length, clip_dur, trim_elapsed)

        # ---------- 6. 人声分离（含“这段没人声”检测与自动换段重试） ----------
        progress(0.3, desc="人声分离")
        vocals, sep_elapsed = yield from _sep_stage(task_id, clip, sep_dir, sep_model, progress)
        vocal_db = audio_rms_db(vocals)
        logger.info(
            "[%s] 人声电平 %.1f dBFS（静音阈值 %.1f，偏轻阈值 %.1f）",
            task_id, vocal_db, VOICE_SILENT_DB, QUIET_VOICE_DB)
        # 多段评估：记录各次尝试，最后取“人声最响”的一段（重试更差不会反而变差）
        best = {"db": vocal_db, "vocals": vocals, "start": start, "length": length,
                "clip_source": clip_source, "clip_dur": clip_dur}
        tried = [(start, start + length)]
        retry_no = 0
        while best["db"] < QUIET_VOICE_DB and retry_no < SEP_SILENT_RETRY and not has_explicit:
            retry_no += 1
            logger.warning(
                "[%s] 该段人声偏轻/几乎没人声（%.1f dBFS），换一段再试（第 %d 次）",
                task_id, best["db"], retry_no)
            a0 = max(0.0, min(t[0] for t in tried) - length)
            a1 = max(t[1] for t in tried) + length
            alt_start, alt_len, alt_src = pick_chorus_window(
                total, target, src, task_id,
                avoid_range=(a0, a1),
                max_start=max(0.0, total * 0.85 - target))
            if alt_len <= 0:
                break
            tried.append((alt_start, alt_start + alt_len))
            clip_r = task_dir / f"clip_r{retry_no}.wav"
            ffmpeg_trim(src, alt_start, alt_len, clip_r)
            clip_dur_r = probe_duration(clip_r)
            # 每次重试独立子目录，避免 rglob 拿到上一轮的 Vocals
            sep_dir_retry = sep_dir / f"retry{retry_no}"
            sep_dir_retry.mkdir(parents=True, exist_ok=True)
            vocals_r, sep_elapsed_r = yield from _sep_stage(
                task_id, clip_r, sep_dir_retry, sep_model, progress)
            sep_elapsed = float(sep_elapsed or 0) + float(sep_elapsed_r or 0)
            db_r = audio_rms_db(vocals_r)
            logger.info(
                "[%s] 重试片段 %.2fs ~ %.2fs 人声 %.1f dBFS",
                task_id, alt_start, alt_start + alt_len, db_r)
            if db_r > best["db"]:
                best = {"db": db_r, "vocals": vocals_r, "start": alt_start,
                        "length": alt_len, "clip_source": "retry_" + alt_src,
                        "clip_dur": clip_dur_r}
        if best["db"] < VOICE_SILENT_DB:
            raise GatewayError(
                f"这段音频几乎没有人声（电平 {best['db']:.0f} dBFS，可能正好是间奏/尾奏），"
                "换个版本或指定时间段再试")
        if best["vocals"] is not vocals:
            logger.info(
                "[%s] 采用人声最响的一段：%.2fs ~ %.2fs（%.1f dBFS，%s）",
                task_id, best["start"], best["start"] + best["length"],
                best["db"], best["clip_source"])
            vocals = best["vocals"]
            start, length = best["start"], best["length"]
            clip_source, clip_dur = best["clip_source"], best["clip_dur"]

        # ---------- 7. RVC 转换 ----------
        progress(0.6, desc="RVC 推理")
        params = {
            "model": model, "index_path": index_path, "pitch": pitch,
            "f0_method": f0, "index_rate": effective_index_rate,
            "index_rate_requested": idx_rate, "protect": protect,
            "resample_sr": resample_sr, "rms_mix_rate": rms_mix,
            "output_format": fmt, "speaker_id": sid,
        }
        final = OUT_DIR / f"{task_id}.{fmt}"
        rvc_elapsed, used_index = yield from run_rvc_stage(task_id, vocals, final, params, progress)

        # ---------- 8. 人声音量调整（vocal_volume_db，0dB 时跳过） ----------
        gain_elapsed = 0.0
        if abs(vocal_gain) >= 0.01:
            t_g = time.monotonic()
            gained = OUT_DIR / f"{task_id}.gain.{fmt}"
            ffmpeg_apply_gain(final, vocal_gain, gained)
            try:
                final.unlink()
            except OSError:
                pass
            final = gained
            gain_elapsed = time.monotonic() - t_g
            logger.info("[%s] 应用人声音量 %.2fdB，耗时 %.2fs", task_id, vocal_gain, gain_elapsed)

        # ---------- 8.5 精挑最终输出区间（仅 chorus：在候选片段内取「目标时长」） ----------
        final_window = None
        pre_trim_dur = probe_duration(final)
        if rmode == "chorus" and requested > 0 and pre_trim_dur > requested + 0.5:
            t_f = time.monotonic()
            # CUSTOM-4.2：最终长度不固定，允许在 [目标, 目标+缓冲] 内取最优片段
            _max_len = min(requested + SEP_BUFFER_SECONDS, MAX_CLIP_SECONDS, pre_trim_dur)
            _min_len = min(requested, _max_len)
            f_start, f_len, f_src = pick_chorus_window(
                pre_trim_dur, _max_len, final, task_id, min_target=_min_len)
            if f_len > 0 and (f_start > 0.05 or f_len < pre_trim_dur - 0.05):
                picked = OUT_DIR / f"{task_id}.picked.{fmt}"
                ffmpeg_trim_fmt(final, f_start, f_len, picked, fmt)
                try:
                    final.unlink()
                except OSError:
                    pass
                final = picked
                final_window = {
                    "start": round(f_start, 3), "length": round(f_len, 3),
                    "source": f_src, "candidate_duration": round(pre_trim_dur, 3),
                }
                logger.info(
                    "[%s] 最终区间精挑：候选 %.2fs -> 输出 %.2fs（目标 %.1f~%.1f，起点 %.2fs，依据=%s，耗时 %.2fs）",
                    task_id, pre_trim_dur, f_len, _min_len, _max_len,
                    f_start, f_src, time.monotonic() - t_f)
            else:
                logger.info("[%s] 候选片段 %.2fs 无需精挑，直接输出", task_id, pre_trim_dur)
        elif rmode == "chorus":
            logger.info(
                "[%s] 候选片段 %.2fs 不足目标 %.1fs（源音频限制），按实际长度输出，不伪造时长",
                task_id, pre_trim_dur, requested)

        # ---------- 8.6 首尾淡入淡出（CUSTOM-5） ----------
        if fade > 0.01:
            _fade_dur = probe_duration(final)
            _fd = min(fade, max(0.05, _fade_dur / 2.0))
            faded = OUT_DIR / f"{task_id}.fade.{fmt}"
            t_fade = time.monotonic()
            ffmpeg_apply_fade(final, _fd, _fd, faded, fmt, _fade_dur)
            try:
                final.unlink()
            except OSError:
                pass
            final = faded
            logger.info("[%s] 首尾淡入淡出 %.2fs（输出 %.2fs），耗时 %.2fs",
                        task_id, _fd, _fade_dur, time.monotonic() - t_fade)

        # ---------- 8.7 人声混响（CUSTOM-7 卷积混响；legacy aecho 回声默认关） ----------
        reverb_info = {"enabled": bool(reverb_on), "applied": False,
                       "wet": round(rev_wet, 3), "dry": round(rev_dry, 3),
                       "decay": round(rev_decay, 3)}
        if reverb_on or echo_on:
            try:
                rev_file = OUT_DIR / f"{task_id}_reverb.{fmt}"
                if reverb_on:
                    rev_elapsed, rev_used = apply_reverb(
                        final, rev_file, fmt, rev_wet, rev_dry,
                        rev_room, rev_decay, rev_damp, rev_pre, rev_width)
                    reverb_info["engine"] = "conv_ir"
                    reverb_info["params"] = rev_used
                else:
                    rev_elapsed, rev_used = apply_echo(
                        final, rev_file, fmt, echo_wet_v, rev_dry, echo_decay_v)
                    reverb_info["engine"] = "aecho_legacy"
                    reverb_info["params"] = rev_used
                final = rev_file
                reverb_info.update({"applied": True, "elapsed": rev_elapsed,
                                    "output": rev_file.name})
                logger.info("[%s] 人声混响完成（%s）：耗时 %.2fs -> %s",
                            task_id, reverb_info["engine"], rev_elapsed, rev_file.name)
            except GatewayError as exc:
                # 混响失败不伪造成功：记录错误并保留原始 RVC 输出
                reverb_info["error"] = str(exc)
                logger.error("[%s] 人声混响失败，保留原始输出: %s", task_id, exc)

        # ---------- 9. 输出校验 ----------
        if not final.is_file() or final.stat().st_size == 0:
            raise GatewayError("RVC 未产出有效文件")
        out_dur = probe_duration(final)
        os.chmod(final, 0o644)
        host_path = to_host_path(str(final))

        record = {
            "task_id": task_id,
            "source": str(song_name_src),
            "source_container_path": str(src),
            "source_duration": round(total, 3),
            "clip_start": round(start, 3),
            "clip_length": round(length, 3),
            "clip_duration": round(clip_dur, 3),
            "clip_source": clip_source,
            "clip_target": round(target, 3),
            "sep_buffer_seconds": SEP_BUFFER_SECONDS,
            "reverb": reverb_info,
            "fade_seconds": round(fade, 2),
            "final_window": final_window,
            "render_mode": rmode,
            "chorus_seconds_requested": round(requested, 3),
            "vocal_volume_db": vocal_gain,
            "inst_volume_db": inst_volume_db,
            "model": model,
            "index": used_index,
            "sep_model": sep_model,
            "pitch": pitch, "f0_method": f0,
            "index_rate": effective_index_rate, "index_rate_requested": idx_rate,
            "protect": protect, "resample_sr": resample_sr, "rms_mix_rate": rms_mix,
            "output_format": fmt, "speaker_id": sid,
            "output": str(final), "host_output": host_path,
            "output_duration": round(out_dur, 3),
            "timings": {
                "queue_wait": round(waited, 2),
                "trim": round(trim_elapsed, 2),
                "separate": round(sep_elapsed, 2),
                "rvc": round(rvc_elapsed, 2),
                "gain": round(gain_elapsed, 2),
            },
            "unsupported_params": unsupported,
        }
        (LOG_DIR / f"{task_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        (CACHE_DIR / f"{key}.json").write_text(
            json.dumps({"output": str(final), "host_output": host_path, **record},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%s] 完成 -> %s（输出 %.2fs）", task_id, host_path, out_dur)

        progress(1.0, desc="完成")
        yield host_path
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] 未预期错误", task_id)
        raise GatewayError(f"网关内部错误: {type(exc).__name__}: {exc}") from exc
    finally:
        _kill_procs(task_id)
        _cleanup_task_dir(task_dir)
        if acquired:
            _TASK_SLOT.release()


# --------------------------------------------------------------------------- #
# TTS 语音转换（CUSTOM-7）：RVC 变声 + 可选卷积混响，无裁剪、无分离、无缓存
# --------------------------------------------------------------------------- #

def sweep_tts_tasks(ttl_seconds: int | None = None) -> int:
    """清理过期的 TTS 任务目录（只处理 TTS_ROOT 下超过 TTL 的，绝不动 out/模型）。"""
    ttl = TTS_TTL_SECONDS if ttl_seconds is None else max(60, int(ttl_seconds))
    if not TTS_ROOT.is_dir():
        return 0
    removed = 0
    now = time.time()
    for d in TTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        try:
            if now - d.stat().st_mtime > ttl:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            pass
    if removed:
        logger.info("清理过期 TTS 任务目录 %d 个（TTL=%ds）", removed, ttl)
    return removed


def resolve_tts_source(raw: str) -> Path:
    """校验 TTS 输入路径：只接受 /work/tts 与 /work/uploads 下的文件（白名单）。"""
    raw = str(raw or "").strip().strip('"').strip("'")
    if not raw:
        raise GatewayError("tts_src 为空")
    candidate = Path(to_container_path(raw))
    resolved = candidate.resolve() if candidate.exists() else candidate
    work = Path(CONTAINER_WORK_DIR).resolve()
    allowed_roots = (work / "tts", work / "uploads")
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise GatewayError(
            f"TTS 输入路径不在允许目录内: {raw!r}"
            f"（只接受 {HOST_WORK_DIR}/tts 与 {HOST_WORK_DIR}/uploads 下的文件）")
    if not resolved.is_file():
        raise GatewayError(f"TTS 输入文件不存在: {resolved}")
    return resolved


def tts_convert(
    tts_src,
    model_dropdown="",
    key_shift=0,
    f0_method="",
    index_rate="",
    protect=None,
    rms_mix_rate=None,
    resample_sr=0,
    output_format="",
    speaker_id=0,
    reverb_enabled=False,
    reverb_wet=REVERB_WET_DEFAULT,
    reverb_dry=REVERB_DRY_DEFAULT,
    reverb_decay=REVERB_DECAY_DEFAULT,
    reverb_room_size=REVERB_ROOM_DEFAULT,
    reverb_damping=REVERB_DAMPING_DEFAULT,
    reverb_pre_delay=REVERB_PREDELAY_DEFAULT,
    reverb_width=REVERB_WIDTH_DEFAULT,
    progress=gr.Progress(),
):
    """把一段（TTS 生成的）语音用 RVC 换声，返回宿主机可读取的成品 JSON。

    与 /convert 共用同一个全局串行信号量：同一时间只跑一个推理任务（歌曲或 TTS）。
    无缓存：TTS 输入是文本级、重复率低，缓存命中率不值得占空间。
    """
    task_id = uuid.uuid4().hex[:12]
    task_dir = TTS_ROOT / task_id
    acquired = False
    try:
        _ensure_dirs()
        sweep_tts_tasks()
        # 与 /convert 共用全局串行信号量：其他任务未完成时**等待**而非拒绝
        waited = 0.0
        while not _TASK_SLOT.acquire(blocking=False):
            waited += 0.5
            if waited > _env_float("GW_SLOT_WAIT", 180.0):
                raise GatewayError("已有推理任务在执行且等待超时，请稍后再试（网关全局串行）。")
            progress(0.02, desc="等待其他任务完成")
            yield None  # 本端点输出是 JSON，进度只走 gr.Progress()
            time.sleep(0.5)
        acquired = True

        # ---------- 1. 输入校验 ----------
        src = resolve_tts_source(tts_src)
        size = src.stat().st_size
        if size > TTS_MAX_BYTES:
            raise GatewayError(f"TTS 输入文件过大: {size / 1048576:.1f}MB（上限 {TTS_MAX_BYTES / 1048576:.0f}MB）")
        total = probe_duration(src)
        if total > TTS_MAX_SECONDS:
            raise GatewayError(f"TTS 音频过长: {total:.1f}s（上限 {TTS_MAX_SECONDS:.0f}s）")

        available = list_rvc_models()
        if not available:
            raise GatewayError(f"音色目录为空: {WEIGHTS_DIR}")
        model = str(model_dropdown or "").strip()
        if not model:
            model = available[0]
        if model not in available:
            raise GatewayError(f"音色模型不存在: {model}（可用: {', '.join(available)}）")

        pitch = _as_int(key_shift, DEFAULT_PITCH, "key_shift", -24, 24)
        f0 = _as_choice(f0_method, DEFAULT_F0_METHOD, "f0_method", ALLOWED_F0)
        idx_rate = _as_float(index_rate, DEFAULT_INDEX_RATE, "index_rate", 0.0, 1.0)
        protect_v = _as_float(protect, DEFAULT_PROTECT, "protect", 0.0, 0.5)
        rms_mix = _as_float(rms_mix_rate, 0.35, "rms_mix_rate", 0.0, 1.0)
        resample = _as_int(resample_sr, DEFAULT_RESAMPLE_SR, "resample_sr", 0, 48000)
        fmt = _as_choice(output_format, "wav", "output_format", ALLOWED_FORMATS)
        sid = _as_int(speaker_id, 0, "speaker_id", 0, 4096)
        reverb_on = _as_bool(reverb_enabled, False)
        rev_wet = _as_float(reverb_wet, REVERB_WET_DEFAULT, "reverb_wet", 0.0, 0.5)
        rev_dry = _as_float(reverb_dry, REVERB_DRY_DEFAULT, "reverb_dry", 0.5, 1.0)
        rev_decay = _as_float(reverb_decay, REVERB_DECAY_DEFAULT, "reverb_decay", 0.2, 4.0)
        rev_room = _as_float(reverb_room_size, REVERB_ROOM_DEFAULT, "reverb_room_size", 0.0, 1.0)
        rev_damp = _as_float(reverb_damping, REVERB_DAMPING_DEFAULT, "reverb_damping", 0.0, 1.0)
        rev_pre = _as_float(reverb_pre_delay, REVERB_PREDELAY_DEFAULT, "reverb_pre_delay", 0.0, 100.0)
        rev_width = _as_float(reverb_width, REVERB_WIDTH_DEFAULT, "reverb_width", 0.0, 1.0)

        index_path = find_index_for(model)
        if index_path:
            effective_index_rate = idx_rate
        else:
            effective_index_rate = 0.0
            logger.warning("[%s] TTS 音色 %s 未找到配对 index，无索引推理（index_rate %.2f -> 0.0）",
                           task_id, model, idx_rate)

        # ---------- 2. 统一转 44.1k 单声道 wav（TTS 原始音频可能是 mp3/ogg/任意采样率） ----------
        progress(0.1, desc="准备音频")
        started = time.monotonic()
        task_dir.mkdir(parents=True, exist_ok=True)
        in_wav = task_dir / "in_44k.wav"
        pre_cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                   "-i", str(src), "-ac", "1", "-ar", str(REVERB_SR),
                   "-c:a", "pcm_s16le", str(in_wav)]
        pre_proc = subprocess.run(pre_cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
        if pre_proc.returncode != 0 or not in_wav.is_file() or in_wav.stat().st_size == 0:
            raise GatewayError(f"TTS 输入转码失败: {pre_proc.stderr.strip()[-300:]}")

        # ---------- 3. RVC 推理（与 /convert 共用 _rvc_stage / run_stage） ----------
        progress(0.4, desc="RVC 推理")
        params = {
            "model": model, "index_path": index_path, "pitch": pitch,
            "f0_method": f0, "index_rate": effective_index_rate,
            "protect": protect_v, "resample_sr": resample,
            "rms_mix_rate": rms_mix, "output_format": fmt, "speaker_id": sid,
        }
        final = OUT_DIR / f"{task_id}_tts.{fmt}"
        # 注意：本端点输出组件是 gr.JSON，_rvc_stage/run_stage 中间 yield 的
        # 进度字符串（"RVC推理中（12s）"）不能透传，否则 JSON 组件解析失败；
        # 进度统一走 gr.Progress()，中间消息在这里吞掉（换 None 跳过更新）。
        stage_gen = run_rvc_stage(task_id, in_wav, final, params, progress)
        while True:
            try:
                next(stage_gen)
                yield None
            except StopIteration as stop:
                rvc_elapsed, used_index = stop.value
                break

        # ---------- 4. 可选卷积混响 ----------
        reverb_info = {"enabled": bool(reverb_on), "applied": False}
        if reverb_on:
            try:
                rev_file = OUT_DIR / f"{task_id}_tts_reverb.{fmt}"
                rev_elapsed, rev_used = apply_reverb(
                    final, rev_file, fmt, rev_wet, rev_dry,
                    rev_room, rev_decay, rev_damp, rev_pre, rev_width)
                final = rev_file
                reverb_info.update({"applied": True, "engine": "conv_ir",
                                    "params": rev_used, "elapsed": rev_elapsed})
            except GatewayError as exc:
                reverb_info["error"] = str(exc)
                logger.error("[%s] TTS 混响失败，保留原始输出: %s", task_id, exc)

        # ---------- 5. 输出校验 ----------
        if not final.is_file() or final.stat().st_size == 0:
            raise GatewayError("TTS 转换未产出有效文件")
        out_dur = probe_duration(final)
        os.chmod(final, 0o644)
        host_path = to_host_path(str(final))
        elapsed = round(time.monotonic() - started, 2)

        record = {
            "task_id": task_id, "kind": "tts",
            "source": str(tts_src), "source_duration": round(total, 3),
            "model": model, "index": used_index,
            "pitch": pitch, "f0_method": f0,
            "index_rate": effective_index_rate, "index_rate_requested": idx_rate,
            "protect": protect_v, "rms_mix_rate": rms_mix,
            "resample_sr": resample, "output_format": fmt, "speaker_id": sid,
            "reverb": reverb_info,
            "output": str(final), "host_output": host_path,
            "output_duration": round(out_dur, 3),
            "elapsed": elapsed,
            "queue_wait": round(waited, 2),
            "rvc_elapsed": round(rvc_elapsed, 2),
        }
        (LOG_DIR / f"{task_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("[%s] TTS 转换完成 -> %s（输出 %.2fs，总耗时 %.1fs）",
                    task_id, host_path, out_dur, elapsed)
        progress(1.0, desc="完成")
        yield record
    except GatewayError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("[%s] TTS 未预期错误", task_id)
        raise GatewayError(f"TTS 网关内部错误: {type(exc).__name__}: {exc}") from exc
    finally:
        _kill_procs(task_id)
        if task_dir.exists():
            shutil.rmtree(task_dir, ignore_errors=True)
        if acquired:
            _TASK_SLOT.release()


def select_msst_model(model_name: str) -> dict:
    """切换/校验分离模型：必须真实存在才返回 success。"""
    name = str(model_name or "").strip()
    if not name:
        return {"success": False, "error": "模型名为空"}
    try:
        sep_model_path(name)
    except GatewayError as exc:
        return {"success": False, "error": str(exc)}
    # 注意：不要回传容器内的绝对路径字符串 —— gradio_client 会把它当成可下载文件去抓取并 403。
    return {"success": True, "selected": name}


def show_msst_models() -> list:
    models = list_sep_models()
    default = SEP_MODEL_DEFAULT
    ordered = [m for m in models if m["id"] == default] + \
              [m for m in models if m["id"] != default]
    return ordered


def show_model() -> list:
    return list_rvc_models()


def model_info(models_dir: str = "") -> dict:
    """chorus-mw 0.9.0 契约：音色模型目录自检（插件 /刷新rvc模型 优先用它）。"""
    backend = WEIGHTS_DIR
    models = list_rvc_models()
    idx_dir = Path(INDEX_DIR)
    info = {
        "dir": backend,
        "backend_dir": backend,
        "source": "RVC Gateway 挂载的音色目录（只读）",
        "exists": Path(backend).is_dir(),
        "count": len(models),
        "models": models,
        # CUSTOM-7：每个模型的 index 匹配情况（模型管理页显示）
        "index_matches": {m: bool(find_index_for(m)) for m in models},
        "matches_backend": True,
        "index_dirs": [str(idx_dir)] if idx_dir.is_dir() else [],
        "rvc_webui_running": False,
        "note": "本网关不走 RVC WebUI，直接调用 infer/cli.py",
    }
    check = str(models_dir or "").strip()
    if check:
        p = Path(to_container_path(check))
        info["checked_dir"] = check
        info["checked_container_path"] = str(p)
        info["checked_exists"] = p.is_dir()
        info["checked_models"] = sorted(x.name for x in p.glob("*.pth")) if p.is_dir() else []
        info["matches_backend"] = str(p) == backend
    if not models:
        info["warning"] = f"音色目录里没有 .pth：{backend}"
    return info


# --------------------------------------------------------------------------- #
# Gradio 应用
# --------------------------------------------------------------------------- #

def build_app() -> gr.Blocks:
    with gr.Blocks(title="RVC Gateway") as demo:
        # convert 返回纯字符串路径 -> Textbox
        out = gr.Textbox(label="result", show_label=True)
        # 列表/字典必须原样到达插件（插件会 isinstance(dict/list) 校验）
        # Textbox 会把它们 str() 掉，因此这些端点统一走 JSON 输出
        out_json = gr.JSON(label="json result")

        i_song = gr.Textbox(value="", label="song_name_src")
        i_key = gr.Number(value=0, label="key_shift", precision=0)
        i_vvol = gr.Number(value=0, label="vocal_vol")
        i_ivol = gr.Number(value=0, label="inst_vol")
        i_model = gr.Textbox(value="", label="model_dropdown")
        i_rev = gr.Number(value=0, label="reverb_intensity")
        i_delay = gr.Number(value=0, label="delay_intensity")
        i_agg = gr.Number(value=10, label="uvr5_agg", precision=0)
        i_tta = gr.Checkbox(value=False, label="uvr5_tta")
        i_pp = gr.Checkbox(value=False, label="uvr5_postprocess")
        i_win = gr.Number(value=512, label="uvr5_window_size", precision=0)
        i_hep = gr.Textbox(value="mirroring", label="uvr5_high_end_process")
        i_mb = gr.Number(value=1, label="msst_batch_size", precision=0)
        i_mo = gr.Number(value=4, label="msst_num_overlap", precision=0)
        i_mn = gr.Checkbox(value=False, label="msst_normalize")
        i_vpp = gr.Checkbox(value=False, label="vocal_postprocess")
        i_sa = gr.Checkbox(value=True, label="shift_accompaniment")
        i_f0 = gr.Textbox(value=DEFAULT_F0_METHOD, label="f0_method")
        i_ir = gr.Textbox(value=str(DEFAULT_INDEX_RATE), label="index_rate")
        i_fr = gr.Number(value=3, label="filter_radius", precision=0)
        i_mm = gr.Textbox(value=SEP_MODEL_DEFAULT, label="msst_model")
        i_mt = gr.Checkbox(value=False, label="msst_use_tta")
        i_st = gr.Textbox(value="", label="start_time")
        i_et = gr.Textbox(value="", label="end_time")
        i_du = gr.Textbox(value="", label="duration")
        i_of = gr.Textbox(value="", label="output_format")
        i_sid = gr.Number(value=0, label="speaker_id", precision=0)
        i_scope = gr.Textbox(value="all", label="scope")
        i_mdir = gr.Textbox(value="", label="models_dir")
        # chorus-mw 契约扩展参数
        i_rmode = gr.Textbox(value="chorus", label="render_mode")
        i_csec = gr.Number(value=DEFAULT_CLIP_SECONDS, label="chorus_seconds", precision=0)
        i_rms = gr.Number(value=DEFAULT_RMS_MIX_RATE, label="rms_mix_rate")
        i_prot = gr.Number(value=DEFAULT_PROTECT, label="protect")
        i_instv = gr.Number(value=-6.0, label="inst_volume_db")
        i_vocv = gr.Number(value=0.0, label="vocal_volume_db")
        i_fade = gr.Number(value=0.0, label="fade_seconds")
        i_rev_on = gr.Checkbox(value=False, label="reverb_enabled")
        i_rev_wet = gr.Number(value=REVERB_WET_DEFAULT, label="reverb_wet")
        i_rev_dry = gr.Number(value=REVERB_DRY_DEFAULT, label="reverb_dry")
        i_rev_decay = gr.Number(value=REVERB_DECAY_DEFAULT, label="reverb_decay")
        i_rev_room = gr.Number(value=REVERB_ROOM_DEFAULT, label="reverb_room_size")
        i_rev_damp = gr.Number(value=REVERB_DAMPING_DEFAULT, label="reverb_damping")
        i_rev_pred = gr.Number(value=REVERB_PREDELAY_DEFAULT, label="reverb_pre_delay")
        i_rev_width = gr.Number(value=REVERB_WIDTH_DEFAULT, label="reverb_width")
        i_echo_on = gr.Checkbox(value=False, label="echo_enabled")
        i_echo_wet = gr.Number(value=ECHO_WET_DEFAULT, label="echo_wet")
        i_echo_decay = gr.Number(value=ECHO_DECAY_DEFAULT, label="echo_decay")
        i_sep_song = gr.Textbox(value="", label="separate_src")
        i_sep_model = gr.Textbox(value="", label="separate_model")
        i_sep_mode = gr.Radio(choices=["both", "vocals", "instrumental"], value="both", label="separate_mode")
        i_sep_fmt = gr.Radio(choices=["wav", "flac", "mp3", "m4a"], value="wav", label="separate_format")
        i_clip_song = gr.Textbox(value="", label="clip_src")
        i_clip_sec = gr.Number(value=25, label="clip_seconds", precision=0)
        i_clip_fmt = gr.Radio(choices=["wav", "flac", "mp3", "m4a"], value="wav", label="clip_format")
        out_clip = gr.JSON(label="clip result")
        out_sep = gr.JSON(label="separate result")
        # ---- CUSTOM-7：TTS 语音转换 ----
        i_tts_src = gr.Textbox(value="", label="tts_src")
        i_tts_model = gr.Textbox(value="", label="tts_model")
        i_tts_key = gr.Number(value=0, label="tts_key_shift", precision=0)
        i_tts_f0 = gr.Textbox(value=DEFAULT_F0_METHOD, label="tts_f0_method")
        i_tts_ir = gr.Textbox(value=str(DEFAULT_INDEX_RATE), label="tts_index_rate")
        i_tts_prot = gr.Number(value=DEFAULT_PROTECT, label="tts_protect")
        i_tts_rms = gr.Number(value=0.35, label="tts_rms_mix_rate")
        i_tts_rs = gr.Number(value=0, label="tts_resample_sr", precision=0)
        i_tts_fmt = gr.Radio(choices=["wav", "flac", "mp3", "m4a"], value="wav", label="tts_format")
        i_tts_sid = gr.Number(value=0, label="tts_speaker_id", precision=0)
        out_tts = gr.JSON(label="tts result")

        ins = [i_song, i_key, i_vvol, i_ivol, i_model, i_rev, i_delay, i_agg, i_tta,
               i_pp, i_win, i_hep, i_mb, i_mo, i_mn, i_vpp, i_sa, i_f0, i_ir, i_fr,
               i_mm, i_mt, i_st, i_et, i_du, i_of, i_sid,
               i_rmode, i_csec, i_rms, i_prot, i_instv, i_vocv, i_fade,
               i_rev_on, i_rev_wet, i_rev_dry, i_rev_decay,
               i_rev_room, i_rev_damp, i_rev_pred, i_rev_width,
               i_echo_on, i_echo_wet, i_echo_decay]

        gr.Button("convert").click(convert, inputs=ins, outputs=out, api_name="convert")
        gr.Button("models").click(show_model, inputs=None, outputs=out_json, api_name="show_model")
        gr.Button("msst").click(show_msst_models, inputs=None, outputs=out_json, api_name="show_msst_models")
        gr.Button("clip").click(
            clip_audio,
            inputs=[i_clip_song, i_clip_sec, i_clip_fmt],
            outputs=out_clip,
            api_name="clip",
        )
        gr.Button("separate").click(
            separate_audio,
            inputs=[i_sep_song, i_sep_model, i_sep_fmt, i_sep_mode],
            outputs=out_sep,
            api_name="separate",
        )
        gr.Button("sel").click(select_msst_model, inputs=[i_mm], outputs=out_json, api_name="select_msst_model")
        gr.Button("cache").click(cache_info, inputs=None, outputs=out_json, api_name="cache_info")
        gr.Button("clr").click(clear_cache, inputs=[i_scope], outputs=out_json, api_name="clear_cache")
        gr.Button("minfo").click(model_info, inputs=[i_mdir], outputs=out_json, api_name="model_info")
        gr.Button("tts").click(
            tts_convert,
            inputs=[i_tts_src, i_tts_model, i_tts_key, i_tts_f0, i_tts_ir,
                    i_tts_prot, i_tts_rms, i_tts_rs, i_tts_fmt, i_tts_sid,
                    i_rev_on, i_rev_wet, i_rev_dry, i_rev_decay,
                    i_rev_room, i_rev_damp, i_rev_pred, i_rev_width],
            outputs=out_tts,
            api_name="tts_convert",
        )
    return demo


def _validate_startup_paths() -> None:
    """启动前检查必要配置与路径，缺失时给出明确错误并退出（CUSTOM-9）。"""
    problems = []
    if not HOST_WORK_DIR:
        problems.append("缺少环境变量 GW_HOST_WORK_DIR（网关工作目录的宿主机路径，"
                        "由 compose/.env 提供）")
    for label, path in (
        ("音色模型目录 GW_WEIGHTS_DIR", WEIGHTS_DIR),
        ("index 目录 GW_INDEX_DIR", INDEX_DIR),
        ("分离模型目录 GW_PYMSS_MODEL_DIR", PYMSS_MODEL_DIR),
    ):
        p = Path(path)
        if not p.is_dir():
            problems.append(f"{label} 不存在或未挂载: {path}")
        elif label.startswith("音色") and not list(p.glob("*.pth")):
            problems.append(f"{label} 里没有任何 .pth 模型: {path}")
    if not Path(SYSTEM_PYTHON).exists():
        problems.append(f"系统 Python 不存在: {SYSTEM_PYTHON}")
    if problems:
        for item in problems:
            logger.error("[启动检查] %s", item)
        raise SystemExit(
            "网关启动检查失败（共 %d 项）。请检查 compose/.env 的路径配置与挂载。" % len(problems))


if __name__ == "__main__":
    _validate_startup_paths()
    _ensure_dirs()
    sweep_stale_tasks(ttl_seconds=0)  # 启动时清掉上次被中断留下的任务残骸
    sweep_tts_tasks(ttl_seconds=0)
    logger.info(
        "Gateway 启动: weights=%s index=%s sep=%s work=%s(->%s) max_clip=%.0fs",
        WEIGHTS_DIR, INDEX_DIR, PYMSS_MODEL_DIR, CONTAINER_WORK_DIR, HOST_WORK_DIR, MAX_CLIP_SECONDS,
    )
    logger.info("可用音色: %s", list_rvc_models())
    logger.info("可用分离模型: %s", [m["id"] for m in list_sep_models()])
    build_app().queue(default_concurrency_limit=1, max_size=QUEUE_MAX_SIZE).launch(
        server_name="0.0.0.0",
        server_port=int(_env("GW_PORT", "7865")),
        # 必须开启：否则拒绝原因（片段超限/音源不可读等）不会回传给插件
        show_error=True,
    )
