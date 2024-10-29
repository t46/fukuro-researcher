#!/bin/bash

# nohup で ollama をバックグラウンドで実行
nohup ollama serve > /ollama.log 2>&1 &

# 現在のシェルに入る
exec bash