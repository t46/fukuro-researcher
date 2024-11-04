import os
import sys
import datasets
import re

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import run_llm


def extract_tokenize_dataset_function(response):
    pattern = r'def tokenize_dataset\(.*?^    return.*?$'
    match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
    if match:
        tokenize_dataset_function_str = match.group(0)
        return tokenize_dataset_function_str
    else:
        return None

def generate_tokenize_dataset_func(dataset_sample):

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
                    targets = batch['targets'].to(device)

                    outputs = model(input_ids, attention_mask=attention_mask, targets=targets)
    ```

    Generate a `tokenize_dataset` function that make the above code work:

    def tokenize_dataset(dataset: datasets.Dataset, tokenizer, tokenizer_max_length: int) -> datasets.Dataset:
        def tokenize_function(examples):
            # encode input
            ... = examples["..."]
            encodings = tokenizer(..., truncation=True, padding="max_length", max_length=tokenizer_max_length)
            # encode target
            ... = examples["..."]
            encodings_target = tokenizer(..., truncation=True, padding="max_length", max_length=tokenizer_max_length)

            encoding_dict = {{
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'targets': encodings_target['input_ids']
            }}

            return encoding_dict

        return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    """

    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )

    tokenize_dataset_function_str = extract_tokenize_dataset_function(response)

    namespace = {
        'datasets': datasets,
        'Dataset': datasets.Dataset
    }

    exec(tokenize_dataset_function_str, namespace)

    tokenize_dataset = namespace['tokenize_dataset']

    return tokenize_dataset
