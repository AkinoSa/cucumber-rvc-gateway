# cucumber-rvc-gateway · RVC 控制台网关

<div align="center">

**AstrBot 插件「RVC 控制台」的独立推理网关**

取音源 → 时长校验 → FFmpeg 裁剪 → 3HP 人声分离(PyMSS) → RVC 推理 → 卷积混响 → 返回宿主机可读路径
另含 TTS 语音换声接口(`/tts_convert`)与任务全局串行调度

![Moe Counter](https://count.getloli.com/get/@cucumber_rvc_gateway?theme=rule34)

</div>

配套插件:**[astrbot_plugin_cucumber_rvc_console · RVC 控制台](https://github.com/AkinoSa/astrbot_plugin_cucumber_rvc_console)**(聊天交互、任务调度、消息发送都在插件侧;网关只做推理)。

## 架构

```
AstrBot 插件(宿主机进程)
   │ ①写 TTS 原始音频 → gateway_work/tts/<task_id>/in.wav
   │ ②Gradio 调用 ──────────────► 本网关(全局串行队列)
   │                                  ├─ FFmpeg 裁剪(严格时间校验)
   │                                  ├─ 3HP 人声分离(PyMSS 子进程)
   │                                  ├─ RVC 推理(infer/cli.py 子进程 / 常驻 worker)
   │                                  ├─ 卷积混响(合成 IR + FFT)
   │ ③读成品路径 ◄────────────────── ┴─ out/<task_id>.wav(宿主机可读)
```

- **纯 CPU 推理**(Intel x86_64 实测;GPU/ARM 未验证,不承诺可用)
- **单实例设计**:全局串行锁不跨进程,**不要启动第二个实例**
- 不修改基础 RVC 镜像的系统 Python;Gradio 隔离在 /opt/gwvenv;推理子进程用系统 Python
- 模型文件**不打包进镜像**,通过挂载提供;新增音色模型无需重建镜像

## 系统要求

| 项 | 要求 |
|---|---|
| 系统 | Linux x86_64 + Docker(实测飞牛 fnOS;Debian/Ubuntu 理论兼容未逐一验证) |
| CPU | 4 核起(实测 i5-6500) |
| 内存 | 16GB 物理内存(容器上限 8~12g;推理峰值 ~4GiB) |
| GPU | 无需(CPU 推理;GPU 未验证) |
| 基础镜像 | RVC 整合包构建的 `rvc-project-rvc:latest`(含 torch/transformers/FFmpeg,见下文「从源码构建」) |

## Docker Compose 部署(推荐)

```bash
git clone https://github.com/AkinoSa/cucumber-rvc-gateway.git
cd cucumber-rvc-gateway
cp .env.example .env
# 编辑 .env:填模型目录、工作目录、AstrBot plugin_data 路径与端口
docker compose up -d --build
curl http://127.0.0.1:3333/   # 返回页面即启动成功
```

### 从源码构建(备用)

`docker compose up -d --build` 本身就是从本目录源码构建;如需脱离 compose 手动构建:`docker build -t cucumber-rvc-gateway:latest .`。**GHCR 预构建镜像尚未发布**,发布后此处会更新为 `docker pull ghcr.io/AkinoSa/cucumber-rvc-gateway`。

## .env 配置说明

| 变量 | 必填 | 说明 |
|---|---|---|
| `RVC_WEIGHTS_DIR` | ✅ | 音色模型 `.pth` 目录(只读挂载到 /app/assets/weights) |
| `RVC_INDICES_DIR` | ✅ | 特征 index 目录(按模型名自动匹配,缺失时无索引推理) |
| `RVC_PYMSS_WEIGHTS_DIR` | ✅ | 3HP 分离模型目录(递归扫描) |
| `RVC_HUBERT_DIR` / `RVC_RMVPE_DIR` | ✅ | HuBERT / RMVPE 模型目录 |
| `RVC_WORK_DIR` | ✅ | 工作目录(cache/out/logs/tasks/tts/uploads),读写 |
| `RVC_PLUGIN_DATA_DIR` | ✅ | AstrBot 的 `data/plugin_data` 宿主机路径(插件下载的歌曲供网关读取) |
| `RVC_GATEWAY_PORT` | | 宿主机端口(默认 3333,可改) |
| `RVC_CPU_LIMIT` / `RVC_MEMORY_LIMIT` / `RVC_MEMORY_SWAP_LIMIT` | | 资源限制(默认 3 核 / 8g / 12g) |
| `RVC_SEP_MODEL` | | 分离模型文件名(默认 `3_HP-Vocal-UVR.pth`,须存在于 PYMSS 目录) |
| `RVC_MAX_CLIP_SECONDS` / `RVC_OMP_THREADS` | | 最大片段(默认 30s)/ 推理线程(默认 2,4 线程易 OOM) |
| `RVC_NETEASE_BASE` / `TZ` | | 备用音源 API / 时区 |

> `.env` 含本机路径,**不要提交到公开仓库**(已在 .gitignore)。

## 模型目录准备

| 挂载目标(容器内) | 内容 | 来源 |
|---|---|---|
| /app/assets/weights | 音色模型 `*.pth` | RVC 训练产出;放入即可被动态发现 |
| /app/assets/indices | 特征索引 `*.index` | 可选;按模型名自动匹配 |
| /app/assets/pymss_weights | 3HP 分离模型(可含子目录,递归扫描) | PyMSS/MSST 系 |
| /app/assets/hubert_base | HuBERT 模型 | RVC 整合包 |
| /app/assets/rmvpe | RMVPE 音高模型 | RVC 整合包 |

模型文件不打包进镜像;新增音色模型放入 weights 目录即被动态发现,无需重启网关。

## API 端点(Gradio named endpoints)

| 端点 | 说明 |
|---|---|
| `/convert` | 歌曲翻唱:取源→裁剪→分离→RVC→淡入淡出→混响;`render_mode`/`chorus_seconds`/`start_time` 等参数 |
| `/tts_convert` | TTS 音频换声(输入限 /work/tts、/work/uploads 白名单;时长≤60s) |
| `/show_model` / `/model_info` | 音色列表 / 目录自检(含 index_matches) |
| `/separate` / `/clip` | 仅分离 / 仅切段 |
| `/show_msst_models` / `/select_msst_model` | 分离模型查看/切换 |
| `/cache_info` / `/clear_cache` | 缓存 |

参数按名字校验;插件传了不支持的参数会被记录而非报错;返回值为宿主机可读路径。

## 健康检查与启动自检

- compose 内置 healthcheck(python 探测 7865 端口);
- 启动时 `_validate_startup_paths()`:模型/index/分离模型/系统 Python 任一缺失 → 明确报错退出(不静默回退);
- `curl http://127.0.0.1:3333/` 返回 200 即在线。

## 模型常驻与加载机制

- **常驻 worker**:`rvc_worker.py` 保持音色模型在内存(首次任务加载 ~40s,后续免加载);空闲 10 分钟自动卸载;
- worker 失败/超时自动回退子进程推理(`infer/cli.py`),两者使用同一套推理实现;
- 3HP 分离每任务以子进程运行,结束释放;
- 全局串行:同一时间只有一个推理任务,其他任务等待(≤180s,超时明确报错);等待耗时记录于任务日志 `queue_wait`。

## 日志与性能监控

- 任务日志:`<WORK_DIR>/logs/<task_id>.json`(阶段耗时:裁剪/分离/推理/混响/排队等待);
- 容器日志:`docker logs cucumber-rvc-gateway`;
- 实测参考(i5-6500 CPU):30s 歌曲裁剪 0.5s → 分离 ~70s → RVC ~54s(热)→ 混响 0.2s。

## 更新与回滚

- **更新**:`git pull` → `docker compose up -d --build`(工作目录与模型在宿主机,不受影响);
- **回滚**:恢复 gateway.py 为备份版本 → `docker compose up -d --build`;工作目录数据不回滚;
- **备份**:`.env` + compose.yaml + gateway.py 即全部配置;工作目录可整目录备份(不含模型)。

## 常见错误

| 现象 | 原因/处理 |
|---|---|
| 启动即退出,报「启动检查失败」 | .env 路径不存在/未挂载,按报错逐项核对 |
| 「音色模型不存在: X」 | weights 目录没有该 .pth,或文件名不带 .pth |
| 「已有推理任务在执行且等待超时」 | 全局串行,前任务超 180s;确认前任务状态或重启网关 |
| 退出码 -9 | OOM;缩短片段或提高内存上限 |
| AstrBot 读不到输出文件 | 网关 `.env` 的 `RVC_WORK_DIR` 与插件设置的「网关工作目录」不一致 |

## 许可证与插件项目

- 本项目以 **AGPL-3.0** 发布(见 [LICENSE](LICENSE));配套插件同样为 AGPL-3.0;
- 推理能力基于 [Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)(MIT)与 [PyMSS/MSST](https://github.com/SUC-DriverOld/MSST-WebUI) 系模型,均以子进程/挂载方式使用;
- 插件项目:[astrbot_plugin_cucumber_rvc_console](https://github.com/AkinoSa/astrbot_plugin_cucumber_rvc_console)。
