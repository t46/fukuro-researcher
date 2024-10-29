"""
Given the dataset, model, and tokenizer, we execute both the Algorithm and the modified NewAlgorithm, and compare and evaluate the results using compare_and_evaluate_algorithms.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import get_linear_schedule_with_warmup
from algorithm import Algorithm
import csv

def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class NewAlgorithm(Algorithm):
    def __init__(self, model, tokenizer, device, tokenize_dataset):
        super().__init__(model, tokenizer, device, tokenize_dataset)
    
    # implement the proposed idea in override methods of Algorithm
    ...

def compare_and_evaluate_algorithms(log, new_log, test_dataset):
    # implement the way to compare the log of Algorithm and NewAlgorithm to evaluate the effectiveness of the proposed method
    ...

    results = {
        "baseline": ...,
        "proposal": ...,
    }

    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", ...])
        for algorithm in results:
            writer.writerow([algorithm, ...])

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_preparation import prepare_model
    from tokenize_dataset_func_generator import generate_tokenize_dataset_func
    from datasets import load_from_disk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workspace_directory = os.path.dirname(os.path.abspath(__file__))
    dataset = load_from_disk(os.path.join(workspace_directory, "dataset"))

    model, tokenizer = prepare_model("google/gemma-2-2b-it", is_pretrained=True)
    tokenize_dataset = generate_tokenize_dataset_func(dataset_sample=dataset["train"][0])

    algorithm = Algorithm(model, tokenizer, device, tokenize_dataset)
    log = algorithm(dataset, is_train_included=False)

    new_algorithm = NewAlgorithm(model, tokenizer, device, tokenize_dataset)
    new_log = new_algorithm(dataset, is_train_included=False)

    compare_and_evaluate_algorithms(log, new_log, dataset["test"])

    print("Finished!!")
