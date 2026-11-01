# NOTICE

## 项目来源

cucumber-rvc-gateway 是 [astrbot_plugin_cucumber_rvc_console(RVC 控制台)](https://github.com/AkinoSa/astrbot_plugin_cucumber_rvc_console) 的配套推理网关,为 AstrBot AI 翻唱/TTS 语音回复提供音源处理、人声分离与 RVC 推理服务。

## 第三方组件

| 组件 | 许可证 | 使用方式 |
|---|---|---|
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT | 基础镜像含其推理代码;子进程调用 infer/cli.py |
| [PyMSS / MSST-WebUI](https://github.com/SUC-DriverOld/MSST-WebUI) | GPL-3.0(模型与代码按其声明) | 子进程调用 3HP 人声分离 |
| [gradio](https://github.com/gradio-app/gradio) | Apache-2.0 | 网关 API 服务 |
| [FFmpeg](https://ffmpeg.org) | LGPL/GPL(按构建) | 裁剪/转码 |

## RVC 模型与 3HP 分离模型

模型文件不随本仓库分发,由使用者自行准备并通过挂载提供。使用者须自行确认所使用模型与音频内容的授权与合规性。

## 许可证

本项目以 GNU AGPL-3.0 发布(见 LICENSE)。
