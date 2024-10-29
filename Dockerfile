# Python 3.11の公式軽量イメージを使用
FROM python:3.11-slim

# 必要なシステムツールとビルドツールのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /root

# ローカルのファイルをコンテナにコピー
COPY . /root/

# ollamaのインストール
RUN curl -fsSL https://ollama.com/install.sh | sh
# RUN nohup ollama serve > ollama.log 2>&1 & \
#     sleep 10 && \
#     ollama pull gemma2

RUN pip install --no-cache-dir --upgrade pip && pip install uv

RUN uv sync

# pipのアップグレードとPythonパッケージのインストール
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir \
#         aider-chat \
#         anthropic \
#         backoff \
#         datasets \
#         huggingface-hub \
#         jupyter \
#         matplotlib \
#         mlcroissant \
#         numpy \
#         ollama \
#         openai \
#         prompt2model \
#         pymupdf4llm \
#         pypdf \
#         tiktoken \
#         timm \
#         torch \
#         tqdm \
#         transformers \
#         vllm \
#         wandb

# スクリプトに実行権限を付与
RUN chmod +x /root/entrypoint.sh

# コンテナ起動時のデフォルトコマンドをbashに設定
CMD ["/root/entrypoint.sh"]