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
    visible_file_names = [f"{workspace_directory}/p2m_experiment.py"]
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
    Large language models (LLMs) have demonstrated remarkable capabilities across various natural language tasks, yet their ability to perform complex logical reasoning, particularly in zero-shot scenarios, remains a significant challenge. While these models can effectively handle pattern matching and statistical inference, they often struggle with structured logical deduction and systematic reasoning. This limitation becomes particularly evident when models encounter novel problem structures or need to apply logical principles across different domains. Current approaches typically rely heavily on extensive fine-tuning or elaborate prompt engineering to achieve acceptable performance in logical reasoning tasks, making them impractical for real-world applications that require flexible and robust reasoning capabilities.
    Recent studies have highlighted a crucial gap between the statistical learning paradigm that underlies current LLMs and the kind of systematic logical reasoning that humans naturally employ. This disconnect manifests in several ways: inconsistent performance across similar logical problems, difficulty in providing transparent explanations for their reasoning steps, and limited ability to transfer logical principles from one context to another. These challenges are compounded by the fact that most existing approaches to improving logical reasoning capabilities focus on enhancing pattern recognition through larger datasets or more sophisticated architectures, rather than addressing the fundamental need for structured logical thinking.
    Furthermore, the opacity of current models' reasoning processes poses a significant barrier to their practical application in fields requiring reliable logical inference. Without clear insight into how models arrive at their conclusions, it becomes difficult to validate their reasoning or identify potential logical fallacies. This lack of transparency not only limits the models' utility in critical applications but also hampers our ability to systematically improve their reasoning capabilities.
    """

    proposition_idea = """
    To address these challenges, we propose a novel Meta-Cognitive Verification Framework (MCVF) that enhances zero-shot logical reasoning capabilities in LLMs. The core innovation of MCVF lies in its introduction of a dedicated meta-cognitive layer that actively monitors and verifies the model's reasoning process in real-time. This layer operates as a separate but integrated component within the model architecture, specifically designed to perform three crucial functions: logical consistency checking, uncertainty quantification, and reasoning path validation.
    The meta-cognitive layer implements a novel attention mechanism that tracks logical dependencies between different reasoning steps. As the model processes a logical problem, this mechanism constructs a dynamic dependency graph that represents the relationships between premises, intermediate conclusions, and final deductions. This graph structure enables the model to perform continuous consistency checks, identifying potential contradictions or gaps in the reasoning chain before they propagate to the final conclusion.
    A key feature of MCVF is its ability to quantify uncertainty at each step of the reasoning process. Unlike traditional approaches that simply provide a final confidence score, our framework generates granular uncertainty metrics for each intermediate logical step. This is achieved through a specialized uncertainty estimation module that combines Bayesian neural networks with symbolic logic principles. The module assesses both aleatoric uncertainty (inherent in the problem structure) and epistemic uncertainty (stemming from the model's knowledge limitations) to provide a comprehensive understanding of the reliability of each reasoning step.
    To implement this framework, we introduce a novel architecture that integrates transformer-based neural networks with symbolic reasoning components. The transformer backbone handles the initial processing of natural language input, while the meta-cognitive layer operates on a more abstract level, working with formalized representations of logical relationships. This hybrid approach allows the model to leverage the strengths of both neural and symbolic processing, creating a more robust and interpretable reasoning system.
    We complement this architectural innovation with a specialized training methodology that emphasizes the development of meta-cognitive awareness. The training process utilizes a curriculum of increasingly complex logical problems, where the model is explicitly trained to identify and correct its own reasoning errors. This includes exposure to adversarial examples specifically designed to trigger logical fallacies, helping the model develop more robust verification capabilities.
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