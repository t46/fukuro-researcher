from datasets import load_dataset

selected_dataset = "wis-k/instruction-following-eval"  # squad
dataset = load_dataset(selected_dataset)
dataset_sample = dataset["train"][0]
print(dataset_sample)

from modules.utils import run_llm

prompt = f"""
You are a helpful assistant.

The dataset is huggingface datasets.Dataset. 
The first element of the dataset is like this:
{dataset_sample}

Model and tokenizer is prepared as:
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    padding_side="left",
    model_max_length=512
)
You will pass this dataset to this model after tokenizing using the tokenizer.

Current code:
```python
        train_dataset = tokenize_dataset(training_datasets, self.tokenizer, self.tokenizer.model_max_length)

        epochs = 1

        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

        # Set up optimizer and scheduler
        optimizer_name = "AdamW"
        optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=5e-3, weight_decay=0.01)  # NOTE: lr を 5e-3 に変更
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_fn = torch.nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {{epoch + 1}}/{{epochs}}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
```
Generate a `tokenize_dataset` function that make the above code work.
"""

response = run_llm(
    model_name="gemma2:9b",
    message=prompt
)
print(response)