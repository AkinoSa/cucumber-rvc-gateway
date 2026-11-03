# NOTICE

## 项目来源

cucumber-rvc-gateway 是 [astrbot_plugin_cucumber_rvc_console(RVC 控制台)](https://github.com/AkinoSa/astrbot_plugin_cucumber_rvc_console) 的配套推理网关,为 AstrBot AI 翻唱/TTS 语音回复提供音源处理、人声分离与 RVC 推理服务。

## 第三方组件

| 组件 | 许可证 | 使用方式 |
|---|---|---|
| [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT (Copyright 2023 liujing04) | 推理代码(commit 81eed5e),子进程调用 infer/cli.py |
| [PyMSS](https://github.com/KitsuneX07/subpixel) | MIT (Copyright 2025 KitsuneX07) | 3HP 人声分离,子进程调用 tools/pymss/cli.py |
| pymss_core | MIT (Copyright 2025 KitsuneX07) | PyMSS 核心库,RVC 仓库内 tools/pymss_core/ |
| [gradio](https://github.com/gradio-app/gradio) | Apache-2.0 | 网关 API 服务(/opt/gwvenv) |
| [FFmpeg](https://ffmpeg.org) | LGPL-2.1+ | 裁剪/转码(Debian 动态链接) |
| [PyTorch](https://pytorch.org) | BSD-3-Clause | CPU 推理(2.4.1+cpu) |
| [transformers](https://huggingface.co/docs/transformers) | Apache-2.0 | HuBERT 加载(4.49.0) |

> 以上所有组件均为 MIT/Apache/BSD 等宽松许可,无 GPL 传染义务,允许商业再分发。

## RVC 模型与 3HP 分离模型

模型文件不随本仓库分发,由使用者自行准备并通过挂载提供。使用者须自行确认所使用模型与音频内容的授权与合规性。

## 许可证

本项目以 GNU AGPL-3.0 发布(见 LICENSE),与插件共享同一许可。
