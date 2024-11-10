"""
Given the dataset, model, and tokenizer, we execute both the MLWorkflow and the modified NewMLWorkflow, and compare and evaluate the results using compare_and_evaluate_proposition.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import get_linear_schedule_with_warmup
from mlworkflow import MLWorkflow
import csv

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    targets = torch.tensor([item['targets'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'targets': targets
    }

class NewMLWorkflow(MLWorkflow):
    def __init__(self, model, tokenizer, device, tokenize_dataset):
        super().__init__(model, tokenizer, device, tokenize_dataset)
        self.structured_prompt_template = """
        Given the following question:
        {question}
        
        Please reason step-by-step to answer the question:
        
        Step 1: {step1}
        Step 2: {step2}
        Step 3: {step3}
        
        Final answer: 
        """

    def run_model(self, test_dataset: datasets.Dataset):
        import time
        start_time = time.time()

        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for item in tqdm(test_dataset, desc="Evaluating"):
                if 'turns' not in item:
                    raise ValueError(f"Required 'turns' column not found. Available columns: {item.keys()}")
                
                question = item['turns'][0][0]  # Extract the question from the 'turns' column
                
                # Iterative prompting
                for i in range(3):  # 3 iterations
                    prompt = self.structured_prompt_template.format(
                        question=question,
                        step1="" if i == 0 else all_outputs[-1].split("\n")[0],
                        step2="" if i < 1 else all_outputs[-1].split("\n")[1],
                        step3="" if i < 2 else all_outputs[-1].split("\n")[2]
                    )
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    all_outputs.append(generated_text)

        self.log["inference_time"] = time.time() - start_time
        self.log["generated_outputs"] = all_outputs
        return self.log

def compare_and_evaluate_proposition(baseline_log, new_log, test_dataset):
    baseline_outputs = baseline_log["generated_outputs"]
    new_outputs = new_log["generated_outputs"]

    # Evaluate logical consistency
    baseline_consistency = evaluate_logical_consistency(baseline_outputs)
    new_consistency = evaluate_logical_consistency(new_outputs)

    # Evaluate step-by-step reasoning
    baseline_steps = count_reasoning_steps(baseline_outputs)
    new_steps = count_reasoning_steps(new_outputs)

    # Evaluate inference time
    baseline_time = baseline_log["inference_time"]
    new_time = new_log["inference_time"]

    results = {
        "baseline": {
            "logical_consistency": baseline_consistency,
            "reasoning_steps": baseline_steps,
            "inference_time": baseline_time
        },
        "proposal": {
            "logical_consistency": new_consistency,
            "reasoning_steps": new_steps,
            "inference_time": new_time
        }
    }

    print("Evaluation Results:")
    print(f"Baseline Logical Consistency: {baseline_consistency:.4f}")
    print(f"Proposal Logical Consistency: {new_consistency:.4f}")
    if baseline_consistency > 0:
        consistency_improvement = (new_consistency - baseline_consistency) / baseline_consistency * 100
        print(f"Logical Consistency Improvement: {consistency_improvement:.2f}%")
    else:
        print("Logical Consistency Improvement: Cannot calculate (baseline is zero)")

    print(f"Baseline Reasoning Steps: {baseline_steps:.2f}")
    print(f"Proposal Reasoning Steps: {new_steps:.2f}")
    if baseline_steps > 0:
        steps_improvement = (new_steps - baseline_steps) / baseline_steps * 100
        print(f"Reasoning Steps Improvement: {steps_improvement:.2f}%")
    else:
        print("Reasoning Steps Improvement: Cannot calculate (baseline is zero)")

    time_change = (new_time - baseline_time) / baseline_time * 100
    print(f"Inference Time Change: {time_change:.2f}%")

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mlworkflow", "logical_consistency", "reasoning_steps", "inference_time"])
        for mlworkflow, data in results.items():
            writer.writerow([mlworkflow, data["logical_consistency"], data["reasoning_steps"], data["inference_time"]])

    return results

def evaluate_logical_consistency(outputs):
    # Implement a simple heuristic for logical consistency
    consistent_count = sum(1 for output in outputs if "therefore" in output.lower() and "because" in output.lower())
    return consistent_count / len(outputs)

def count_reasoning_steps(outputs):
    # Count the average number of reasoning steps
    step_counts = [output.count("Step") for output in outputs]
    return sum(step_counts) / len(step_counts)

def tokenize_dataset(dataset: datasets.Dataset, tokenizer, tokenizer_max_length: int) -> datasets.Dataset:
    def tokenize_function(examples):
        # Print the column names and first few entries to debug
        print("Dataset columns:", examples.keys())
        for key in examples.keys():
            print(f"First entry of '{key}':", examples[key][0])
        
        # Use the 'turns' column which contains the question
        if 'turns' not in examples:
            raise ValueError(f"Required 'turns' column not found. Available columns: {examples.keys()}")
        
        # Extract the question from the 'turns' column
        questions = [turn[0] for turn in examples['turns']]
        
        # Use 'ground_truth' as targets
        if 'ground_truth' not in examples:
            raise ValueError(f"Required 'ground_truth' column not found. Available columns: {examples.keys()}")
        
        targets = examples['ground_truth']
        
        tokenized_inputs = tokenizer(questions, truncation=True, padding="max_length", max_length=tokenizer_max_length)
        tokenized_targets = tokenizer(targets, truncation=True, padding="max_length", max_length=tokenizer_max_length)
        tokenized_inputs["targets"] = tokenized_targets["input_ids"]

        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    return tokenized_dataset

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_preparation import prepare_model
    from datasets import load_from_disk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workspace_directory = os.path.dirname(os.path.abspath(__file__))
    dataset = load_from_disk(os.path.join(workspace_directory, "dataset"))

    # Print dataset info
    print("Dataset info:")
    print(dataset)
    print("Train dataset columns:", dataset["train"].column_names)
    print("Test dataset columns:", dataset["test"].column_names)

    model, tokenizer = prepare_model("google/gemma-2-2b-it", is_pretrained=True)

    mlworkflow = MLWorkflow(model, tokenizer, device, tokenize_dataset)
    log = mlworkflow(dataset, is_train_included=False)

    new_mlworkflow = NewMLWorkflow(model, tokenizer, device, tokenize_dataset)
    new_log = new_mlworkflow(dataset, is_train_included=False)

    compare_and_evaluate_proposition(log, new_log, dataset["test"])

    print("Finished!!")
