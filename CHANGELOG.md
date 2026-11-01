# Changelog — cucumber-rvc-gateway

## 1.0.0 (2026-11-02)

首个公开发布版本(由开发者私有部署版本 CUSTOM-9 整理而来)。

### 功能

- 歌曲翻唱流水线:取音源(http/网易云 ID/本地路径)→ 严格裁剪校验 → FFmpeg 裁剪 → 3HP(PyMSS)人声分离(静音检测+换段重试)→ RVC 推理 → 淡入淡出 → 卷积混响(合成 IR,8 参数)
- TTS 语音换声:`/tts_convert`(路径白名单、时长/大小上限、统一串行队列)
- 常驻 RVC worker(省模型加载;失败回退子进程;空闲自退)
- 全局串行调度(忙时等待≤180s)+ queue_wait 耗时记录
- 模型动态发现 + index 自动匹配 + 无索引降级
- 扩展端点:/separate /clip /model_info(index_matches)/cache_info/clear_cache
- 启动自检(模型/目录/系统 Python 缺失明确报错)
- Gradio queue + 子进程登记/终止(任务取消)

### 环境

- CPU 推理实测(Intel x86_64);GPU/ARM 未验证
- Docker Compose + .env 配置;健康检查

### 已知行为

- 单实例设计(全局锁不跨进程)
- 推理顺序可能与提交顺序交换(取歌耗时不一致)
- TTS 输入路径限 /work/tts 与 /work/uploads 白名单
