import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Dict, List, Any

import anthropic
import openai
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

from ai_scientist.generate_ideas import generate_ideas, check_idea_novelty
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_writeup import perform_writeup
from ai_scientist.perform_review import perform_review, load_paper

# Constants
NUM_REFLECTIONS = 3
SUPPORTED_MODELS = [
    "claude-3-5-sonnet-20240620",
    "gpt-4o-2024-05-13",
    "deepseek-coder-v2-0724",
    "llama3.1-405b",
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0"
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AI scientist experiments")
    parser.add_argument("--skip-idea-generation", action="store_true", help="Skip idea generation and load existing ideas")
    parser.add_argument("--skip-novelty-check", action="store_true", help="Skip novelty check and use existing ideas")
    parser.add_argument("--experiment", type=str, default="nanoGPT", help="Experiment to run AI Scientist on.")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620", choices=SUPPORTED_MODELS, help="Model to use for AI Scientist.")
    parser.add_argument("--writeup", type=str, default="latex", choices=["latex"], help="What format to use for writeup")
    parser.add_argument("--parallel", type=int, default=0, help="Number of parallel processes to run. 0 for sequential execution.")
    parser.add_argument("--improvement", action="store_true", help="Improve based on reviews.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.")
    parser.add_argument("--num-ideas", type=int, default=50, help="Number of ideas to generate")
    return parser.parse_args()

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
    
    logger.info(f"Using {type(client).__name__} with model {client_model}")
    return client, client_model

def setup_coder(model: str, fnames: List[str], io: InputOutput) -> Coder:
    if model == "hybrid":
        main_model = Model("claude-3-5-sonnet-20240620")
    elif model == "deepseek-coder-v2-0724":
        main_model = Model("deepseek/deepseek-coder")
    elif model == "llama3.1-405b":
        main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
    else:
        main_model = Model(model)
    
    return Coder.create(
        main_model=main_model, fnames=fnames, io=io, stream=False, use_git=False, edit_format="diff"
    )

def do_idea(base_dir: str, results_dir: str, idea: Dict[str, Any], model: str, client: Any, client_model: str, writeup: str, improvement: bool, log_file: bool = False) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = os.path.join(results_dir, idea_name)
    
    try:
        os.makedirs(folder_name, exist_ok=False)
    except FileExistsError:
        logger.error(f"Folder {folder_name} already exists.")
        return False

    shutil.copytree(base_dir, folder_name, dirs_exist_ok=True)
    
    with open(os.path.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    
    exp_file = os.path.join(folder_name, "experiment.py")
    vis_file = os.path.join(folder_name, "plot.py")
    notes = os.path.join(folder_name, "notes.txt")
    
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write(f"## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write(f"Description: Baseline results.\n")
    
    if log_file:
        log_path = os.path.join(folder_name, "log.txt")
        logger.addHandler(logging.FileHandler(log_path))
    
    try:
        logger.info(f"Starting idea: {idea_name}")
        
        logger.info("Starting Experiments")
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
        coder = setup_coder(model, fnames, io)
        success = perform_experiments(idea, folder_name, coder, baseline_results)
        if not success:
            logger.error(f"Experiments failed for idea {idea_name}")
            return False
        
        logger.info("Starting Writeup")
        writeup_file = os.path.join(folder_name, "latex", "template.tex")
        fnames = [exp_file, writeup_file, notes]
        coder = setup_coder(model, fnames, io)
        perform_writeup(idea, folder_name, coder, client, client_model)
        
        logger.info("Starting Review")
        paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
        review = perform_review(
            paper_text,
            model="gpt-4o-2024-05-13",
            client=openai.OpenAI(),
            num_reflections=5,
            num_fs_examples=1,
            num_reviews_ensemble=5,
            temperature=0.1,
        )
        with open(os.path.join(folder_name, "review.txt"), "w") as f:
            json.dump(review, f, indent=4)
        
        return True
    
    except Exception as e:
        logger.exception(f"Failed to evaluate idea {idea_name}: {str(e)}")
        return False

def main():
    args = parse_arguments()
    
    client, client_model = setup_client(args.model)
    
    base_dir = os.path.join("templates", args.experiment)
    results_dir = os.path.join("results", args.experiment)
    
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.num_ideas,
        num_reflections=NUM_REFLECTIONS,
    )
    
    ideas = check_idea_novelty(
        ideas,
        base_dir=base_dir,
        client=client,
        model=client_model,
    )
    
    with open(os.path.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)
    
    novel_ideas = [idea for idea in ideas if idea["novel"]]
    
    for idea in novel_ideas:
        logger.info(f"Processing idea: {idea['Name']}")
        success = do_idea(base_dir, results_dir, idea, args.model, client, client_model, args.writeup, args.improvement)
        logger.info(f"Completed idea: {idea['Name']}, Success: {success}")
    
    logger.info("All ideas evaluated.")

if __name__ == "__main__":
    main()