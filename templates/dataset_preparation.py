"""
This script is used to prepare a dataset for a machine learning model.
It uses the Hugging Face Hub API to search for datasets and the LLM API to rename the splits and features of the dataset.
"""
import sys
import os
import re
import torch
from torchvision import transforms
import multiprocessing

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
    # 文字列から辞書部分を抽出する正規表現
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
    1. Rename features that could be model inputs to "input".
    2. Rename features that could be labels or outputs to "output".
    3. Leave other features unchanged.
    Return a Python dictionary with old names as keys and new names as values.
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    
    # Extract the dictionary from the AI's response
    return process_ai_response(response)

def change_split_name(split_names):
    prompt = f"""
    Given these split names: {', '.join(split_names)}. 
    Rename them according to these rules:
    1. Rename splits related to training to "train".
    2. Rename splits related to testing to "test".
    3. Leave other splits unchanged.
    Return a Python dictionary with old names as keys and new names as values.
    """
    
    response = run_llm(
        model_name="gemma2:9b",
        message=prompt
    )
    return process_ai_response(response)

def rename_dataset_with_ai(dataset):
    split_names, feature_names = get_split_and_feature_names(dataset)
    
    # Get the new names for splits and features using AI
    split_name_map = change_split_name(split_names)
    feature_name_map = change_feature_name(feature_names)
    
    # Create a new DatasetDict
    new_dataset = DatasetDict()
    
    # Rename splits and features
    for old_split_name, new_split_name in split_name_map.items():
        # Rename features
        new_features = {}
        for old_feature_name, new_feature_name in feature_name_map.items():
            new_features[new_feature_name] = dataset[old_split_name][old_feature_name]
        
        # Create a new Dataset with renamed features
        new_split_data = Dataset.from_dict(new_features)
        
        # Add the renamed split to the new DatasetDict
        new_dataset[new_split_name] = new_split_data
    
    return new_dataset, split_name_map, feature_name_map

def prepare_dataset(query):
    results = search_datasets(query)
    selected_dataset = results[0]
    dataset = load_dataset(selected_dataset)
    dataset, _, _ = rename_dataset_with_ai(dataset)
    return dataset

def get_available_gpu_memory():
    """Returns a list of tuples containing GPU index and available memory in GB for all available GPUs."""
    gpu_memory = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            free_memory = (total_memory - allocated_memory) / 1e9  # Convert to GB
            gpu_memory.append((i, free_memory))
    return gpu_memory

def get_optimal_num_proc():
    """Returns the optimal number of processes to use based on available CPU cores."""
    return max(1, multiprocessing.cpu_count() - 1)  # Leave one core free for system processes

def preprocess_dataset(dataset, model=None, tokenizer=None):
    available_gpus = get_available_gpu_memory()
    num_proc = get_optimal_num_proc()

    if available_gpus:
        print(f"Found {len(available_gpus)} GPUs. Using all available GPUs for processing.")
        device = torch.device("cuda")
    else:
        print("No GPUs detected. Using CPU for processing.")
        device = torch.device("cpu")

    print(f"Using {num_proc} CPU cores for parallel processing.")

    # timmモデル用の前処理を追加
    if tokenizer is None and model is not None:  # timmモデルの場合
        # モデルから入力サイズと正規化パラメータを取得
        input_size = model.default_cfg['input_size'][-2:]  # (H, W)
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
        
        transform = transforms.Compose([
            transforms.Resize(input_size),  # モデルに対応する入力サイズ
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)  # モデルに対応する正規化
        ])

        def preprocess_function(examples):
            examples['input'] = [transform(img.convert('RGB')).to(device) for img in examples['input']]
            examples['output'] = torch.tensor(examples['output'], device=device)
            return examples

        dataset = dataset.map(preprocess_function, batched=True, num_proc=num_proc)
        dataset.set_format(type='torch', columns=['input', 'output'])
    
    elif tokenizer is not None:  # トークナイザを使用するモデルの場合
        def tokenize_function(examples):
            return tokenizer(examples['input'], truncation=True, padding=True, return_tensors="pt")

        dataset = dataset.map(tokenize_function, batched=True, num_proc=num_proc)
        
        # GPUが利用可能な場合、データをGPUに移動
        if available_gpus:
            dataset = dataset.map(lambda x: {k: v.to(device) for k, v in x.items()}, batched=True)

    dataset = dataset.cache()
    
    return dataset


if __name__ == "__main__":
    query = input("検索キーワードを入力: ")
    dataset = prepare_dataset(query)
    print(dataset)