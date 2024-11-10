# ML-Control-Experiment

## Quick Start
Set up the environment.
```
cp .env.example .env
```

## Build the container
`docker build -t ml-control-experiment .`

## Run the container
`docker run -it --rm --env-file $(pwd)/.env -v $(pwd):/root ml-control-experiment`

## Operations inside the container

`ollama pull gemma2:9b`

`uv run python run_controlled_experiment.py`

