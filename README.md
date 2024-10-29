`src` の中の `p2m_experiment.py` がメイン。これを AI に書き換えてもらう。`debug_p2m_experiment.py` は、そのままでも動く。

.env に API キーを設定する。

## コンテナの起動
`docker run -it --rm --env-file $(pwd)/.env -v $(pwd):/root fukuro-researcher`

## コンテナ内での操作
`nohup ollama serve > ollama.log 2>&1 &`

`ollama pull gemma2:9b`

`uv sync`

`uv run python run_p2m_exp.py`

`uv run python run_inheritance.py`
