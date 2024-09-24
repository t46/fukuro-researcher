from huggingface_hub import HfApi
from transformers import AutoModel, AutoTokenizer
import timm

def search_models(query, max_results=5):
    api = HfApi()
    results = []
    for model in api.list_models(search=query, sort="downloads", direction=-1):
        results.append(model.id)
        if len(results) == max_results:
            break
    return results

def load_model(model_id):
    if model_id.startswith('timm/'):
        # timmを使用してモデルをロード
        model = timm.create_model(model_id.split('/')[-1], pretrained=True)
        return model, None  # VGG16にはtokenizerが不要なのでNoneを返す
    else:
        # 他のモデルの場合は既存のコードを使用
        model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer

def prepare_model(query):
    results = search_models(query)
    selected_model = results[0]
    model, tokenizer = load_model(selected_model)
    return model, tokenizer

if __name__ == "__main__":
    query = input("検索キーワードを入力: ")
    model, tokenizer = prepare_model(query)
    print(model, tokenizer)