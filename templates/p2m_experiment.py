from prompt2model.utils import api_tools
api_tools.default_api_agent = api_tools.APIAgent(model_name="gpt-4o-mini-2024-07-18")


def prepare_dataset(prompt: str):

    ##### データセットの取得 #####
    from prompt2model.dataset_retriever import DescriptionDatasetRetriever
    from prompt2model.prompt_parser import MockPromptSpec, TaskType
    retriever = DescriptionDatasetRetriever()
    task_type = TaskType.TEXT_GENERATION
    prompt_spec = MockPromptSpec(task_type)
    prompt_spec._instruction = prompt
    retrieved_dataset_dict = retriever.retrieve_dataset_dict(
        prompt_spec, blocklist=[]
    )

    from prompt2model.dataset_processor import TextualizeProcessor
    text_processor = TextualizeProcessor(has_encoder=True)
    text_modified_dataset_dicts = text_processor.process_dataset_lists(
        prompt_spec.instruction,
        [retrieved_dataset_dict["train"]],
        train_proportion=0.7,
        val_proportion=0.1,
        maximum_example_num={"train": 3500, "val": 500, "test": 1000},
    )
    train_datasets = [each["train"] for each in text_modified_dataset_dicts]
    val_datasets = [each["val"] for each in text_modified_dataset_dicts]
    test_datasets = [each["test"] for each in text_modified_dataset_dicts]

    return {"train": train_datasets, "val": val_datasets, "test": test_datasets}

def prepare_model(prompt: str):

    from prompt2model.model_retriever import DescriptionModelRetriever
    from prompt2model.prompt_parser import MockPromptSpec, TaskType

    prompt_spec = MockPromptSpec(TaskType.TEXT_GENERATION)
    prompt_spec._instruction = prompt

    retriever = DescriptionModelRetriever(
        search_index_path="path_to_bm25_search_index.pkl",
        model_descriptions_index_path="path_to_model_info_directory",
        use_bm25=True,
        use_HyDE=True,
    )
    pretrained_model_name = retriever.retrieve(prompt_spec)

    model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name,
        padding_side="left",
        model_max_length=512
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def tokenize_dataset(dataset: datasets.Dataset, tokenizer, tokenizer_max_length: int) -> datasets.Dataset:
    dataset = dataset.shuffle(seed=42)
    
    def tokenize_function(examples):
        inputs = examples["model_input"]
        outputs = examples["model_output"]
        full_texts = [f"{inp} {out}" for inp, out in zip(inputs, outputs)]
        encodings = tokenizer(full_texts, truncation=True, padding=True, max_length=tokenizer_max_length)
        encodings["labels"] = encodings["input_ids"].copy()
        encodings["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in encodings["labels"]]
        return encodings

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)


########################## Causal LM に特化した学習コード (from prompt2model.model_trainer.generate import GenerationModelTrainer) ##########################

import datasets
import torch
import transformers
from datasets import concatenate_datasets
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

def train_model(
    model,
    tokenizer,
    device,
    training_datasets: list[datasets.Dataset] | None = None,
) -> tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizer]:

    model.to(device)

    # Concatenate and tokenize datasets
    train_dataset = concatenate_datasets(training_datasets)
    train_dataset = tokenize_dataset(train_dataset, tokenizer, tokenizer.model_max_length)

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
def inference(model, tokenizer, device, test_dataset: datasets.Dataset):
    model.to(device)
    test_dataset = tokenize_dataset(test_dataset, tokenizer, tokenizer.model_max_length)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            all_outputs.append(outputs.logits.argmax(dim=-1).tolist())

    return all_outputs

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt = "hogehoge"
    dataset = prepare_dataset(prompt)
    model, tokenizer = prepare_model(prompt)
    model, tokenizer = train_model(
        model,
        tokenizer,
        device,
        training_datasets=dataset["train"]
    )
    outputs = inference(model, tokenizer, device, dataset["test"])
