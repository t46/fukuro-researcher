########################## Causal LM に特化した学習コード (from prompt2model.model_trainer.generate import GenerationModelTrainer) ##########################

import torch
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from dataset_preparation import tokenize_dataset
import datasets
import torch.optim as optim

class Algorithm:
    def __init__(self, model, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(device)

    def __call__(self, dataset: datasets.Dataset, is_train_included=False):
        if is_train_included:
            self.model, self.tokenizer = self.train_model(dataset["train"])
        outputs = self.run_inference(dataset["test"])
        return outputs
    
    def train_model(self, training_datasets: list[datasets.Dataset] | None = None):

        # Concatenate and tokenize datasets
        train_dataset = concatenate_datasets(training_datasets)
        train_dataset = tokenize_dataset(train_dataset, self.tokenizer, self.tokenizer.model_max_length)

        epochs = 3

        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        # Set up optimizer and scheduler
        optimizer_name = "AdamW"
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=5e-5, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Average train loss: {avg_train_loss:.4f}")

        # モデルの保存
        model.save_pretrained("path_to_save_model")
        tokenizer.save_pretrained("path_to_save_tokenizer")

        return model, tokenizer

    # モデルによる推論&テストデータによる評価（loss ではなく）
    def run_inference(self, test_dataset: datasets.Dataset):
        def collate_fn(batch):  # TODO: 別の場所に移す
            input_ids = torch.tensor([item['input_ids'] for item in batch])
            attention_mask = torch.tensor([item['attention_mask'] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        test_dataset = tokenize_dataset(test_dataset, self.tokenizer, self.tokenizer.model_max_length)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask)
                all_outputs.extend(outputs.logits.argmax(dim=-1).cpu().tolist())

        return all_outputs
    
class NewAlgorithm(Algorithm):
    def __init__(self, model, tokenizer, device):
        super().__init__(model, tokenizer, device)

    def train_model(self, training_datasets: list[datasets.Dataset] | None = None):
        ...
    
    def run_inference(self, test_dataset: datasets.Dataset):
        ...

def compare_and_evaluate_algorithms(outputs, new_outputs):
    import csv
    # それぞれをあるメトリクスで評価
    ...

    results = {
        "baseline": ...,
        "new": ...,
    }

    # csv に結果を出力
    with open("results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", ...])
        for algorithm in results:
            writer.writerow([algorithm, ...])

if __name__ == "__main__":
    from dataset_preparation import prepare_dataset
    from model_preparation import prepare_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_dataset = "mrpc"
    prompt_model = "gpt2"
    dataset = prepare_dataset(prompt_dataset)
    model, tokenizer = prepare_model(prompt_model)

    algorithm = Algorithm(model, tokenizer, device)
    outputs = algorithm(dataset, is_train_included=False)

    new_algorithm = NewAlgorithm(model, tokenizer, device)
    new_outputs = new_algorithm(dataset, is_train_included=False)

    compare_and_evaluate_algorithms(outputs, new_outputs)

    print("Finished!!")
