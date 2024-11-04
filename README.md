`src` の中の `experiment.py` がメイン。これを AI に書き換えてもらう。

.env に API キーを設定する。

## コンテナの起動
`docker run -it --rm --env-file $(pwd)/.env -v $(pwd):/root fukuro-researcher`

## コンテナ内での操作

`ollama pull gemma2:9b`

`uv sync`

`uv run python run_controlled_experiment.py`
