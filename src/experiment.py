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
    
    # implement the proposed idea in override methods of MLWorkflow
    ...

def compare_and_evaluate_proposition(log, new_log, test_dataset):
    # implement the way to compare the log of MLWorkflow and NewMLWorkflow to evaluate the effectiveness of the proposed method
    ...

    results = {
        "baseline": ...,
        "proposal": ...,
    }

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["mlworkflow", ...])
        for mlworkflow in results:
            writer.writerow([mlworkflow, ...])

def tokenize_dataset(dataset: datasets.Dataset, tokenizer, tokenizer_max_length: int) -> datasets.Dataset:
    def tokenize_function(examples):
        # implement the tokenize_function to convert the dataset into a format suitable for the current research context
        tokenized_inputs = tokenizer(examples[...], truncation=True, padding="max_length", max_length=tokenizer_max_length)
        tokenized_targets = tokenizer(examples[...], truncation=True, padding="max_length", max_length=tokenizer_max_length)
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

    model, tokenizer = prepare_model("google/gemma-2-2b-it", is_pretrained=True)

    mlworkflow = MLWorkflow(model, tokenizer, device, tokenize_dataset)
    log = mlworkflow(dataset, is_train_included=False)

    new_mlworkflow = NewMLWorkflow(model, tokenizer, device, tokenize_dataset)
    new_log = new_mlworkflow(dataset, is_train_included=False)

    compare_and_evaluate_proposition(log, new_log, dataset["test"])

    print("Finished!!")
