import openai
from datasets import DatasetDict, Dataset

# OpenAI APIキーの設定
# 注意: 実際の使用時は環境変数やシークレット管理システムを使用してください
openai.api_key = 'your-api-key-here'

def get_split_and_feature_names(dataset):
    split_names = list(dataset.keys())
    feature_names = list(dataset[split_names[0]].features.keys())
    return split_names, feature_names

def ai_rename(names, type):
    prompt = f"Given these {type} names: {', '.join(names)}. Suggest new, more descriptive names for any that need improvement. Return a Python dictionary with old names as keys and new names as values. If a name doesn't need changing, use the original name as the value."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that suggests improved names for dataset splits and features."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the dictionary from the AI's response
    ai_suggestion = eval(response.choices[0].message['content'])
    return ai_suggestion

def change_split_name(split_names):
    return ai_rename(split_names, "split")

def change_feature_name(feature_names):
    return ai_rename(feature_names, "feature")

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

# Example usage:
original_dataset = DatasetDict({
    'train': Dataset.from_dict({'img': [1, 2, 3], 'lbl': ['a', 'b', 'c']}),
    'val': Dataset.from_dict({'img': [4, 5], 'lbl': ['d', 'e']})
})

print("Original dataset:")
print(original_dataset)

renamed_dataset, split_changes, feature_changes = rename_dataset_with_ai(original_dataset)

print("\nRenamed dataset:")
print(renamed_dataset)
print("\nSplit name changes:", split_changes)
print("Feature name changes:", feature_changes)