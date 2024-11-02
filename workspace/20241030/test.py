"""
Given the dataset, model, and tokenizer, we execute both the Algorithm and the modified NewAlgorithm, and compare and evaluate the results using compare_and_evaluate_algorithms.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import get_linear_schedule_with_warmup
from testalgorithm import Algorithm
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
        self.prompt_history = []
        self.max_prompt_history = 3
    
    def run_model(self, test_dataset: datasets.Dataset):
        import time
        start_time = time.time()

        test_dataset = self.tokenize_dataset(test_dataset, self.tokenizer, self.tokenizer.model_max_length)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(test_loader), desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # Dynamic Prompt Assembly
                dynamic_prompt = self.assemble_dynamic_prompt(input_ids[0])
                
                # Prepend dynamic prompt to input
                input_with_prompt = self.tokenizer.encode(dynamic_prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
                input_with_prompt = torch.cat([input_with_prompt.repeat(input_ids.shape[0], 1), input_ids], dim=1)
                
                # Adjust attention mask
                prompt_attention_mask = torch.ones((input_ids.shape[0], input_with_prompt.shape[1] - input_ids.shape[1]), dtype=torch.long).to(self.device)
                attention_mask_with_prompt = torch.cat([prompt_attention_mask, attention_mask], dim=1)

                outputs = self.model.generate(
                    input_with_prompt,
                    attention_mask=attention_mask_with_prompt,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(generated_texts)

                # Update prompt history
                self.update_prompt_history(generated_texts[0])

                if batch_idx == 5:
                    break

        self.log["inference_time"] = time.time() - start_time
        self.log["generated_outputs"] = all_outputs
        self.log["final_prompt"] = self.prompt_history[-1] if self.prompt_history else ""
        return self.log

    def assemble_dynamic_prompt(self, input_ids):
        base_prompt = "Generate a response based on the following context and previous outputs:"
        context = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        if not self.prompt_history:
            return f"{base_prompt}\nContext: {context}\n"
        
        recent_outputs = "\n".join(self.prompt_history)
        return f"{base_prompt}\nContext: {context}\nRecent outputs:\n{recent_outputs}\n"

    def update_prompt_history(self, generated_text):
        self.prompt_history.append(generated_text)
        if len(self.prompt_history) > self.max_prompt_history:
            self.prompt_history.pop(0)

def compare_and_evaluate_algorithms(log, new_log, test_dataset):
    from rouge import Rouge
    from nltk.translate.bleu_score import corpus_bleu
    import numpy as np

    rouge = Rouge()
    
    # Prepare reference and hypothesis texts
    try:
        reference_texts = [str(example['downloads']) for example in test_dataset]
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Available keys in test_dataset:", test_dataset[0].keys())
        raise

    baseline_texts = log["generated_outputs"]
    dpa_texts = new_log["generated_outputs"]

    print(f"Number of reference texts: {len(reference_texts)}")
    print(f"Number of baseline texts: {len(baseline_texts)}")
    print(f"Number of DPA texts: {len(dpa_texts)}")

    # Ensure all texts are strings
    reference_texts = [str(text) for text in reference_texts]
    baseline_texts = [str(text) for text in baseline_texts]
    dpa_texts = [str(text) for text in dpa_texts]

    # Calculate ROUGE scores
    baseline_rouge = rouge.get_scores(baseline_texts, reference_texts, avg=True)
    dpa_rouge = rouge.get_scores(dpa_texts, reference_texts, avg=True)

    # Calculate BLEU scores
    reference_tokens = [str(text).split() for text in reference_texts]
    baseline_tokens = [str(text).split() for text in baseline_texts]
    dpa_tokens = [str(text).split() for text in dpa_texts]

    baseline_bleu = corpus_bleu([reference_tokens] * len(baseline_tokens), baseline_tokens)
    dpa_bleu = corpus_bleu([reference_tokens] * len(dpa_tokens), dpa_tokens)

    # Calculate average output length
    baseline_avg_length = np.mean([len(text.split()) for text in baseline_texts])
    dpa_avg_length = np.mean([len(text.split()) for text in dpa_texts])

    # Prepare results
    results = {
        "baseline": {
            "rouge-1": baseline_rouge["rouge-1"]["f"],
            "rouge-2": baseline_rouge["rouge-2"]["f"],
            "rouge-l": baseline_rouge["rouge-l"]["f"],
            "bleu": baseline_bleu,
            "avg_length": baseline_avg_length,
            "inference_time": log["inference_time"]
        },
        "proposal": {
            "rouge-1": dpa_rouge["rouge-1"]["f"],
            "rouge-2": dpa_rouge["rouge-2"]["f"],
            "rouge-l": dpa_rouge["rouge-l"]["f"],
            "bleu": dpa_bleu,
            "avg_length": dpa_avg_length,
            "inference_time": new_log["inference_time"]
        }
    }

    # Write results to CSV
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "rouge-1", "rouge-2", "rouge-l", "bleu", "avg_length", "inference_time"])
        for algorithm, metrics in results.items():
            writer.writerow([algorithm, metrics["rouge-1"], metrics["rouge-2"], metrics["rouge-l"], metrics["bleu"], metrics["avg_length"], metrics["inference_time"]])

    return results

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model_preparation import prepare_model
    def generate_tokenize_dataset_func(dataset_sample):
        def tokenize_dataset(examples, tokenizer, max_length):
            def tokenize_function(examples):
                # Check available columns
                available_columns = list(examples.keys())
                
                if "organization" in available_columns and "model_name" in available_columns and "downloads" in available_columns:
                    # Combine organization, model_name, and downloads into a single text
                    combined = [f"Organization: {org}, Model: {model}, Downloads: {downloads}" 
                                for org, model, downloads in zip(examples["organization"], 
                                                                 examples["model_name"], 
                                                                 examples["downloads"])]
                else:
                    raise ValueError(f"Unexpected dataset format. Available columns: {available_columns}")
                
                model_inputs = tokenizer(combined, max_length=max_length, truncation=True, padding="max_length")
                
                # Add labels (assuming the model should predict the 'downloads' column)
                model_inputs["labels"] = examples["downloads"]
                
                return model_inputs
            
            return examples.map(tokenize_function, batched=True, remove_columns=examples.column_names)
        return tokenize_dataset
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
