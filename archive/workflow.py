###### Tokenize Dataset 関数の生成 ######

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

# def tokenize_dataset(... で始まるtokenize_datasetを抜き出す
import re

def extract_tokenize_dataset_function(response):
    # def tokenize_dataset(... で始まる行を探す
    match = re.search(r'def tokenize_dataset\(', response, re.DOTALL)
    if match:
        # 該当の行から def tokenize_dataset(... までの行を取得
        start = match.end()
        # 該当の行から def tokenize_dataset(... までの行を取得
        tokenize_dataset_function_str = response[start:].split('\n')[0]
        return tokenize_dataset_function_str
    else:
        return None

tokenize_dataset_function_str = extract_tokenize_dataset_function(response)

# Create a new namespace
namespace = {}

# Execute the string in the new namespace
exec(tokenize_dataset_function_str, namespace)

# Extract the function from the namespace
tokenize_dataset = namespace['tokenize_dataset']


###### prepare_dataset (あるいは experiment.py) に Tokenize Dataset 関数を追記  ######
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

def setup_coder(coder_llm_name, visible_file_names):
    io = InputOutput(
        yes=True, chat_history_file=f"aider.txt"
    )
    coder_model = Model(coder_llm_name)

    coder = Coder.create(
        main_model=coder_model,
        fnames=visible_file_names,
        io=io,
        stream=False,
        use_git=False,
        edit_format="diff",  # diff にすると、ファイルの中身を変更するときに、変更された部分だけを表示する
    )
    return coder

coder_llm_name = "gemma2:9b"
visible_file_names = ["prepare_dataset.py"]
coder = setup_coder(coder_llm_name, visible_file_names)

prompt = f"""
Add the `tokenize_dataset` function to the `prepare_dataset.py`.

`tokenize_dataset` function is:
{tokenize_dataset_function_str}
"""

coder.chat(prompt)