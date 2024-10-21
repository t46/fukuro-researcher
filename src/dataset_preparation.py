"""
This script is used to prepare a dataset for a machine learning model.
It uses the Hugging Face Hub API to search for datasets and the LLM API to rename the splits and features of the dataset.
"""
import sys
import os
import re
import torch
from torchvision import transforms
import datasets

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
from huggingface_hub import HfApi
from utils import run_llm
from datasets import DatasetDict, Dataset

def search_datasets(query, max_results=5):
    api = HfApi()
    results = []
    for dataset in api.list_datasets(search=query, sort="downloads", direction=-1):
        results.append(dataset.id)
        if len(results) == max_results:
            break
    print(results)
    return results

def get_split_and_feature_names(dataset):
    split_names = list(dataset.keys())
    feature_names = list(dataset[split_names[0]].features.keys())
    return split_names, feature_names

def extract_dict(text):
    dict_pattern = r'\{[^}]+\}'
    match = re.search(dict_pattern, text)
    if match:
        return match.group()
    return None

def evaluate_dict_safely(dict_str):
    if dict_str:
        try:
            return eval(dict_str)
        except:
            print("Error: Unable to evaluate the extracted dictionary.")
    else:
        print("Error: No dictionary found in the response.")
    return {}

def process_ai_response(response):
    if isinstance(response, str):
        dict_str = extract_dict(response)
    else:
        try:
            content = response.choices[0].message['content']
            dict_str = extract_dict(content)
        except AttributeError:
            print("Error: Unexpected response format.")
            return {}
    
    return evaluate_dict_safely(dict_str)

def identify_feature_name(feature_names):
    prompt = f"""
    Given these feature names: {', '.join(feature_names)}. 
    Identify the **single** feature that corresponds to 'input' and 'output', respectively. Don't select multiple features.
    Return a Python dictionary with 'input' and 'output' as keys and the corresponding feature names as values.
    {{
        "input": str,
        "output": str
    }}
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    
    return process_ai_response(response)

def identify_split_name(split_names):
    prompt = f"""
    Given these split names: {', '.join(split_names)} 
    Identify the **single** split that corresponds to 'train' and 'test', respectively. Don't select multiple splits.
    Return a Python dictionary with 'train' and 'test' as keys and the corresponding split names as values.
    {{
        "train": str,
        "test": str
    }}
    If there is only one split or no split corresponding to 'test', please input the same value for 'test' as for 'train'.
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    return process_ai_response(response)

def rename_dataset_with_ai(dataset):

    # NOTE: tokenize_dataset を自動生成することになってので、feature の名前を変更する必要がなくなった
    # split_names, feature_names = get_split_and_feature_names(dataset)

    split_names = list(dataset.keys())
    split_name_map = identify_split_name(split_names)

    # split_name_map["train"] と split_name_map["test"] が同じ場合は、シャッフルした後で
    # dataset["train"] がもとのdataset["train"]の80%、dataset["test"] がもとのdataset["test"]の20%になるように分割する

    # シャッフル
    dataset = dataset.shuffle(seed=42)

    # TODO: train と test の分割とかは llm 使わずにルールーベースでやった方がロバストかもしれない

    new_dataset = DatasetDict()
    if split_name_map["train"] == split_name_map["test"]:
        train_size = int(len(dataset[split_name_map["train"]]) * 0.8)
        test_size = len(dataset[split_name_map["train"]]) - train_size
        train_dataset = dataset[split_name_map["train"]].select(range(train_size))
        test_dataset = dataset[split_name_map["train"]].select(range(train_size, train_size + test_size))
        new_dataset["train"] = train_dataset
        new_dataset["test"] = test_dataset
    else:
        new_dataset["train"] = dataset[split_name_map["train"]]
        new_dataset["test"] = dataset[split_name_map["test"]]


    # feature_name_map = identify_feature_name(feature_names)

    # new_dataset = DatasetDict()

    # もし split_name_map["train"] と split_name_map["test"] が同じ場合は
    # dataset["train"] がもとのdataset["train"]の80%、dataset["test"] がもとのdataset["test"]の20%になるように分割する
    # if split_name_map["train"] == split_name_map["test"]:
    #     train_size = int(len(dataset[split_name_map["train"]]) * 0.8)
    #     test_size = len(dataset[split_name_map["train"]]) - train_size
    #     train_dataset = dataset[split_name_map["train"]].select(range(train_size))
    #     test_dataset = dataset[split_name_map["train"]].select(range(train_size, train_size + test_size))
    #     new_dataset[split_name_map["train"]] = train_dataset
    #     new_dataset[split_name_map["test"]] = test_dataset

    # for split in ["train", "test"]:
    #     original_split = split_name_map[split]
    #     new_data = {
    #         "input": dataset[original_split][feature_name_map["input"]],
    #         "output": dataset[original_split][feature_name_map["output"]]
    #     }
    #     new_dataset[split] = Dataset.from_dict(new_data)

    return new_dataset

def prepare_dataset(query):
    results = search_datasets(query)
    selected_dataset = results[0]
    dataset = load_dataset(selected_dataset)
    dataset = rename_dataset_with_ai(dataset)
    return dataset

def preprocess_dataset(dataset, model=None, tokenizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for processing.")

    def process_images(subset):
        input_size = model.default_cfg['input_size'][-2:]  # (H, W)
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
        
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        def preprocess_function(examples):
            examples['input'] = [transform(img.convert('RGB')).to(device) for img in examples['input']]
            examples['output'] = torch.tensor(examples['output'], device=device)
            return examples

        subset = subset.map(preprocess_function, batched=True)
        subset.set_format(type='torch', columns=['input', 'output'])
        return subset

    def process_text(subset):
        # パディングトークンを設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples['input'] = tokenizer(examples['input'], truncation=True, padding=True, return_tensors="pt")
            examples['output'] = torch.tensor(examples['output'], device=device)
            return examples
        
        subset = subset.map(preprocess_function, batched=True)
        subset.set_format(type='torch', columns=['input', 'output'])
        
        # GPU への転送
        # if torch.cuda.is_available():
        #     subset = subset.map(lambda x: {k: v.to(device) for k, v in x.items()}, batched=True)

        return subset

    # DatasetDict の train と test をそれぞれ処理
    if tokenizer is None and model is not None:  # For timm models
        dataset['train'] = process_images(dataset['train'])
        dataset['test'] = process_images(dataset['test'])
    
    elif tokenizer is not None:  # For tokenizer-based models
        dataset['train'] = process_text(dataset['train'])
        dataset['test'] = process_text(dataset['test'])

    return dataset

def tokenize_dataset(dataset: datasets.Dataset, tokenizer, tokenizer_max_length: int) -> datasets.Dataset:
    dataset = dataset.shuffle(seed=42)
    
    def tokenize_function(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        full_texts = [f"{inp} {out}" for inp, out in zip(inputs, outputs)]
        encodings = tokenizer(full_texts, truncation=True, padding="max_length", max_length=tokenizer_max_length)
        encodings["labels"] = encodings["input_ids"].copy()
        encodings["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in encodings["labels"]]
        return encodings

    return dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

if __name__ == "__main__":
    query = input("検索キーワードを入力: ")
    dataset = prepare_dataset(query)
    print(dataset)