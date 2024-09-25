########################## Causal LM に特化した学習コード (from prompt2model.model_trainer.generate import GenerationModelTrainer) ##########################

import torch
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from p2m_modules.modules import tokenize_dataset
import datasets


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
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
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
        test_dataset = tokenize_dataset(test_dataset, self.tokenizer, self.tokenizer.model_max_length)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                all_outputs.append(outputs.logits.argmax(dim=-1).tolist())

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
    from p2m_modules.modules import prepare_dataset, prepare_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "hogehoge"
    dataset = prepare_dataset(prompt)
    model, tokenizer = prepare_model(prompt)

    algorithm = Algorithm(model, tokenizer, device)
    outputs = algorithm(dataset, is_train_included=False)

    new_algorithm = NewAlgorithm(model, tokenizer, device)
    new_outputs = new_algorithm(dataset, is_train_included=False)

    compare_and_evaluate_algorithms(outputs, new_outputs)
