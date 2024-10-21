# Python 3.11の公式軽量イメージを使用
FROM python:3.11-slim

# 必要なシステムツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# pyproject.toml と README.md をコンテナ内にコピー
COPY pyproject.toml README.md /app/

# pipのアップグレード
RUN pip install --no-cache-dir --upgrade pip

# pyproject.tomlの依存関係をインストール
RUN pip install --no-cache-dir $(grep -oP '(?<=")[^"]+(?=")' pyproject.toml | grep -v '^[0-9]')

# コンテナ起動時のデフォルトコマンドをbashに設定
CMD ["bash"]
