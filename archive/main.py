from modules.hypothesis_generation import generate_hypothesis
from modules.verification_execution import execute_verification
from modules.paper_writing import perform_writeup

import os
import json

from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

import openai
import anthropic

llm_name = "gemma2:9b"
coder_llm_name = "ollama/gemma2:9b" #deepseek/deepseek-coder, "deepseek-coder-v2-0724"  # deepseek-coder-v2-0724 は aider coding benchmark で claude sonnet に続いて2番目の性能、ただお金はかかるっぽい

# このファイルがあるディレクトリの中にある outputs ディレクトリ、なければ作成
output_directory = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_directory, exist_ok=True)

# output_directory を RESEARCH_OUTPUT_PATH に設定
os.environ["RESEARCH_OUTPUT_PATH"] = output_directory

# research_metadata.json の ${RESEARCH_OUTPUT_PATH} を os.environ["RESEARCH_OUTPUT_PATH"] に変更
# すでに変更されている場合は何もしない
with open("modules/research_metadata.json", "r") as f:
    research_metadata = json.load(f)  
    for key, value in research_metadata.items():
        if "${RESEARCH_OUTPUT_PATH}" in value:
            research_metadata[key] = value.replace("${RESEARCH_OUTPUT_PATH}", os.environ["RESEARCH_OUTPUT_PATH"])
    with open("modules/research_metadata.json", "w") as f:
        json.dump(research_metadata, f, indent=4)

latex_directory = os.path.join(output_directory, "latex")
os.makedirs(latex_directory, exist_ok=True)

def setup_coder(coder_llm_name, visible_file_names):
    io = InputOutput(
        yes=True, chat_history_file=f"{output_directory}/aider.txt"
    )
    coder_model = Model(coder_llm_name)

    coder = Coder.create(
        main_model=coder_model,
        fnames=visible_file_names,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",  # diff にすると、ファイルの中身を変更するときに、変更された部分だけを表示する
    )
    return coder

# TODO: client に依存してる部分を liteLLM ベースに変更する
def setup_client(model: str) -> tuple:
    if model == "claude-3-5-sonnet-20240620":
        client_model = model
        client = anthropic.Anthropic()
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        client = anthropic.AnthropicBedrock()
    elif model in ["gpt-4o-2024-05-13", "hybrid"]:
        client_model = "gpt-4o-2024-05-13"
        client = openai.OpenAI()
    elif model == "deepseek-coder-v2-0724":
        client_model = model
        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
    elif model == "llama3.1-405b":
        client_model = "meta-llama/llama-3.1-405b-instruct"
        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1"
        )
    else:
        raise ValueError(f"Model {model} not supported.")

    return client, client_model

def main():
    problem = "Create a novel neural network architecture that surpassess transfromer"
    hypothesis = generate_hypothesis(problem, llm_name=llm_name)

    visible_file_names = [f"{output_directory}/experiment.py", f"{output_directory}/plot.py", f"{output_directory}/notes.txt"]
    coder = setup_coder(coder_llm_name, visible_file_names)
    is_successful_verificaition = execute_verification(problem, hypothesis, llm_name=llm_name, coder=coder, output_directory=output_directory)

    visible_file_names = [f"{latex_directory}/template.tex", f"{output_directory}/experiment.py", f"{output_directory}/notes.txt"]
    coder = setup_coder(coder_llm_name, visible_file_names)
    client, client_model = setup_client("llama3.1-405b")
    perform_writeup(folder_name=output_directory, coder=coder, cite_client=client, cite_model=client_model)


if __name__ == "__main__":
    main()