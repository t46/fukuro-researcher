import shutil
import os
import json
import subprocess
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def copy_source_to_workspace(src, dst):
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"Successfully copied '{src}' to '{dst}'")
    except Exception as e:
        print(f"Error: {e}")

def generate_prompts(research_context, proposition_idea):
    dataset_prompt_template = """
    Research Context: {research_context}
    Proposition Idea: {proposition_idea}

    Your final goal is to implement a Proposition Idea and design experiments to validate its feasibility based on the Research Context. To achieve this, generate search query for obtaining datasets from HuggingFace using an LLM. This prompt should explicitly state the purpose and objectives of the research, what the Proposition Idea is, and which datasets should be retrieved to validate it.

    query will be used as follows:
    from huggingface_hub import HfApi
    api = HfApi()
    results = list(api.list_datasets(search=query, sort="downloads", direction=-1, limit=max_results))

    Note that query input to `search` is a string that will be contained in the returned datasets.

    Generate only one query and the query should be single word.
    For example, query will be a dataset name like "wikitext", "cifar10", "imagenet", "sarcasm", "emotion", etc.

    <query>
    "..."
    </query>
    """

    model_prompt_template = """
    Research Context: {research_context}
    Proposition Idea: {proposition_idea}

    Your final goal is to implement a Proposition Idea and design experiments to validate its feasibility based on the Research Context. To achieve this, generate search queryfor obtaining baseline models from HuggingFace using an LLM. This prompt should explicitly state the purpose and objectives of the research, what the Proposition Idea is, and which models should be retrieved to validate it.

    query will be used as follows:
    from huggingface_hub import HfApi
    api = HfApi()
    results = list(api.list_models(search=query, sort="downloads", direction=-1))

    Note that query input to `search` is a string that will be contained in the returned models.

    Generate only one query and the query should be single word.
    For example, query can will a model name like "resnet", "lstm", "bert", "gpt2", "t5", etc.

    <query>
    "..."
    </query>
    """

    dataset_prompt = dataset_prompt_template.format(research_context=research_context, proposition_idea=proposition_idea)
    model_prompt = model_prompt_template.format(research_context=research_context, proposition_idea=proposition_idea)

    return dataset_prompt, model_prompt

def save_prompts(prompts, filename):
    with open(filename, "w") as f:
        json.dump(prompts, f)

def initialize_coder(workspace_directory, coder_llm_name):
    visible_file_names = [f"{workspace_directory}/p2m_experiment.py", f"{workspace_directory}/algorithm.py"]
    io = InputOutput(yes=True, chat_history_file=f"{workspace_directory}/aider.txt")
    coder_model = Model(coder_llm_name)
    return Coder.create(main_model=coder_model, fnames=visible_file_names, io=io, stream=False, use_git=False, edit_format="diff")

def run_experiment(coder, prompt, workspace_directory, timeout=7200):
    coder.run(prompt)
    cwd = os.path.abspath(workspace_directory)
    command = ["python", "p2m_experiment.py"]
    return subprocess.run(command, cwd=cwd, stderr=subprocess.PIPE, text=True, timeout=timeout)

def main():
    # Configuration
    source_directory = "/root/src"  # "/root/fukuro-researcher/src"
    workspace_directory_base = "/root/workspace"  # "/root/fukuro-researcher/workspace"
    workspace_directory = os.path.join(workspace_directory_base, source_directory.split("/")[-1])
    if not os.path.exists(workspace_directory):
        os.makedirs(workspace_directory)
    coder_llm_name = "ollama/gemma2:9b"  # "claude-3-5-sonnet-20240620"
    max_edit_trials = 10

    # Research context and proposition
    research_context = """
    The hallucination problem in Large Language Models (LLMs) — where models generate inaccurate or non-factual content — is a critical issue, especially in high-stakes fields like healthcare, law, and education. 

    Challenge: Automatic Detection and Evaluation of Hallucinations

    Background: Hallucinated content lacks clear error messages, making manual verification costly and time-consuming.
    """

    proposition_idea = """
    - Consistency Checking Algorithm: Generate multiple outputs and detect inconsistencies among them.
    - Fact-checking with Ensemble Models: Integrate multiple fact-checking engines to cross-validate outputs in real time.
    - Context-aware Error Detection: Automatically identify hallucinations by comparing generated text with pre-defined knowledge bases.
    """

    # Execute steps
    copy_source_to_workspace(source_directory, workspace_directory)
    dataset_prompt, model_prompt = generate_prompts(research_context, proposition_idea)
    prompts_file = os.path.join(workspace_directory, "prompts.json")
    save_prompts({"dataset": dataset_prompt, "model": model_prompt}, prompts_file)

    coder = initialize_coder(workspace_directory, coder_llm_name)

    experiment_coding_prompt = """
    Research Context: {research_context}
    Proposition Idea: {proposition_idea}

    Code Explanation:
    p2m_experiment.py represents the workflow of a machine learning research that verifies the effectiveness of the proposed method through comparative experiments. Specifically, given the dataset, model, and tokenizer, it executes Algorithm and NewAlgorithm (which is a modified version of Algorithm), and then compares and evaluates their results using compare_and_evaluate_algorithms.

    Algorithm represents a typical machine learning workflow where the model is trained on training data and then executed on test data. NewAlgorithm inherits from this and modifies the workflow by overriding train_model, run_model, or both.

    This code embodies the idea that "machine learning research that validates a proposal through comparative experiments is an endeavor to determine whether adding a new proposal (NewAlgorithm) to an Algorithm that generates certain output from data yields better results in an expected sense."

    Task Description:
    Please edit p2m_experiment.py to implement the Proposition Idea and design experiments to validate its feasibility based on the Research Context.
    Your task is to complete the experimental code by editing p2m_experiment.py.
    Please edit the following parts, but do not change any other parts:

    model, tokenizer = prepare_model(prompt_model, is_pretrained=True)
    If you need to train from a randomly initialized state, set is_pretrained=False. If you're fine-tuning a pre-trained model or using a pre-trained model without further training, set is_pretrained=True.

    NewAlgorithm
    To implement the Proposition Idea and design experiments to validate its effectiveness, override one or all methods of Algorithm. For example, if you're proposing a new Optimizer, implement the new optimizer and use it in train_model instead of the existing optimizer. If you're proposing a new neural architecture, implement the new architecture in the Hugging Face format and assign it in the part where self.model = model.to(device) is set in the __init__ method. If you're proposing a prompt technique to improve the zero-shot inference performance of a pre-trained model, implement the prompt technique in the run_model part. In this way, first consider which part of the machine learning workflow the Proposition Idea is addressing, and then implement NewAlgorithm to properly implement and experiment with the proposal. When doing so, make sure to define all the information needed to see if the proposal is superior to existing methods in the expected sense using self.log.

    compare_and_evaluate_algorithms
    Implement evaluation criteria to examine how and in what sense the Proposition Idea is superior to existing methods. For example, if the proposed method is expected to predict better than existing methods, you might compare if the accuracy is higher. Or, if you're proposing an optimization method that's expected to converge faster, you might compare the number of steps it took before the loss reached a certain value. Also, if you're proposing a method with superior interpretability, you might define some metric that shows that the internal representation of the model is more interpretable in some sense and compare that. In this way, consider in what sense the Proposition Idea is expected to be superior to existing methods in relation to the Research Context, and implement evaluation metrics that can compare this.
    """.format(research_context=research_context, proposition_idea=proposition_idea)

    for i in range(max_edit_trials):
        run_result = run_experiment(coder, experiment_coding_prompt, workspace_directory)

        if run_result.returncode == 0:
            print("Successfully executed")
            break
        else:
            print("Failed to execute")
            error_message = run_result.stderr
            print(error_message)
            experiment_coding_prompt = f"error: {error_message}\nprompt: "

        if i == max_edit_trials - 1:
            print("Failed to execute after max trials")
            break

if __name__ == "__main__":
    main()