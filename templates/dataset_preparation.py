"""
This script is used to prepare a dataset for a machine learning model.
It uses the Hugging Face Hub API to search for datasets and the LLM API to rename the splits and features of the dataset.
"""
import sys
import os
import re
import torch
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from huggingface_hub import HfApi
from modules.utils import run_llm
from datasets import DatasetDict, Dataset

def search_datasets(query, max_results=5):
    api = HfApi()
    results = []
    for dataset in api.list_datasets(search=query, sort="downloads", direction=-1):
        results.append(dataset.id)
        if len(results) == max_results:
            break
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

def change_feature_name(feature_names):
    prompt = f"""
    Given these feature names: {', '.join(feature_names)}. 
    Rename them according to these rules:
    0. If "input" and "label" are included, keep them as is.
    1. If "input" is not included, rename only one feature most related to input to "input".
    2. If "label" is not included, rename only one feature most related to label to "label".
    3. Leave other features unchanged.
    Return a Python dictionary with old names as keys and new names as values. label only the dictionary.
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    
    return process_ai_response(response)

def change_split_name(split_names):
    prompt = f"""
    Given these split names: {', '.join(split_names)}. 
    Rename them according to these rules:
    0. If "train" and "test" are included, keep them as is.
    1. If "train" is not included, rename only one split most related to training to "train".
    2. If "test" is not included, rename only one split most related to testing to "test".
    3. Leave other splits unchanged.
    Return a Python dictionary with old names as keys and new names as values. label only the dictionary.
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    return process_ai_response(response)

def rename_dataset_with_ai(dataset):
    split_names, feature_names = get_split_and_feature_names(dataset)
    
    split_name_map = change_split_name(split_names)
    feature_name_map = change_feature_name(feature_names)
    
    new_dataset = DatasetDict()
    
    for old_split_name, new_split_name in split_name_map.items():
        new_features = {}
        for old_feature_name, new_feature_name in feature_name_map.items():
            new_features[new_feature_name] = dataset[old_split_name][old_feature_name]
        
        new_split_data = Dataset.from_dict(new_features)
        new_dataset[new_split_name] = new_split_data

    # 指定されたsplit_nameとfeature_name以外を除外する処理
    allowed_splits = ['train', 'test']
    allowed_features = ['input', 'label']
    
    # split_nameのフィルタリング
    for split_name in list(new_dataset.keys()):
        if split_name not in allowed_splits:
            del new_dataset[split_name]
    
    # feature_nameのフィルタリング
    for split_name in new_dataset:
        new_dataset[split_name] = new_dataset[split_name].select_columns(allowed_features)
    
    return new_dataset, split_name_map, feature_name_map


def prepare_dataset(query):
    results = search_datasets(query)
    selected_dataset = results[0]
    dataset = load_dataset(selected_dataset)
    dataset, _, _ = rename_dataset_with_ai(dataset)
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
            examples['label'] = torch.tensor(examples['label'], device=device)
            return examples

        subset = subset.map(preprocess_function, batched=True)
        subset.set_format(type='torch', columns=['input', 'label'])
        return subset

    def process_text(subset):
        # パディングトークンを設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            examples['input'] = tokenizer(examples['input'], truncation=True, padding=True, return_tensors="pt")
            examples['label'] = torch.tensor(examples['label'], device=device)
            return examples
        
        subset = subset.map(preprocess_function, batched=True)
        subset.set_format(type='torch', columns=['input', 'label'])
        
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

if __name__ == "__main__":
    query = input("検索キーワードを入力: ")
    dataset = prepare_dataset(query)
    print(dataset)