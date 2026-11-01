# RVC Gateway —— 为 AstrBot / RVC 控制台插件 提供中间层
#
# 关键设计：
#   * 不改动基础镜像里的任何系统包（torch / transformers / huggingface-hub 保持原样）
#   * Gradio 装在隔离的 /opt/gwvenv，避免 huggingface-hub 被升级而破坏 transformers 4.49.0
#   * 推理子进程一律用系统 /usr/local/bin/python
FROM rvc-project-rvc:latest

ENV PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GW_APP_DIR=/app \
    GW_CONTAINER_WORK_DIR=/work \
    GW_HOST_WORK_DIR=<网关工作目录> \
    GW_CONTAINER_PLUGIN_DIR=/mnt/plugin \
    GW_HOST_PLUGIN_DIR=<插件目录>

# 隔离虚拟环境：继承系统包（torch 等），但 pip 只写入 venv
RUN python -m venv --system-site-packages /opt/gwvenv \
    && /opt/gwvenv/bin/python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/gwvenv/bin/pip install --no-cache-dir "gradio==6.31.0"

# 校验：系统 Python 的 transformers / HuBERT 依赖必须仍然可用
RUN python -c "import importlib.metadata as m; import transformers; \
    assert m.version('huggingface_hub').startswith('0.'), m.version('huggingface_hub'); \
    from transformers import HubertModel; print('system python OK:', m.version('huggingface_hub'))" \
    && /opt/gwvenv/bin/python -c "import gradio, gradio_client, torch; \
    print('gateway venv OK:', gradio.__version__, gradio_client.__version__, torch.__version__)"

COPY gateway.py rvc_worker.py /gateway/

WORKDIR /app
EXPOSE 7865
CMD ["/opt/gwvenv/bin/python", "/gateway/gateway.py"]
