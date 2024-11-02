import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
from transformers import get_linear_schedule_with_warmup

def collate_fn(batch):  # TODO: 別の場所に移す
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    result = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
    }
    if 'labels' in batch[0]:
        labels = torch.tensor([item['labels'] for item in batch])
        result['labels'] = labels
    return result

class Algorithm:
    def __init__(self, model, tokenizer, device, tokenize_dataset):
        self.tokenizer = tokenizer
        self.device = device
        self.model = model.to(device)
        self.tokenize_dataset = tokenize_dataset
        self.log = {
            "train_losses": [],
            "eval_losses": [],
            "train_time": None,
            "inference_time": None,
            "generated_outputs": [],
            "model_parameters": None,
        }

    def __call__(self, dataset: datasets.Dataset, is_train_included=False):
        if is_train_included:
            self.model, self.tokenizer = self.train_model(dataset["train"])
        log = self.run_model(dataset["test"])
        return log
    
    def train_model(self, training_datasets: list[datasets.Dataset] | None = None):
        import time
        start_time = time.time()

        # Concatenate and tokenize datasets
        # train_dataset = concatenate_datasets(training_datasets)
        train_dataset = self.tokenize_dataset(training_datasets, self.tokenizer, self.tokenizer.model_max_length)

        epochs = 3

        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        # Set up optimizer and scheduler
        optimizer_name = "AdamW"
        optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=5e-5, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Average train loss: {avg_train_loss:.4f}")
            self.log["train_losses"].append(avg_train_loss)

        self.log["train_time"] = time.time() - start_time
        self.log["model_parameters"] = self.model.state_dict()

        # モデルの保存
        self.model.save_pretrained("artifacts")
        self.tokenizer.save_pretrained("artifacts")

        return self.model, self.tokenizer

    # モデルによる推論&テストデータによる評価（loss ではなく）
    def run_model(self, test_dataset: datasets.Dataset):
        import time
        start_time = time.time()

        test_dataset = self.tokenize_dataset(test_dataset, self.tokenizer, self.tokenizer.model_max_length)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

        all_outputs = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(generated_texts)

        self.log["inference_time"] = time.time() - start_time
        self.log["generated_outputs"] = all_outputs
        return self.log
