from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def search_models(query, max_results=5):
    api = HfApi()
    results = []
    for model in api.list_models(search=query, sort="downloads", direction=-1):
        results.append(model.id)
        if len(results) == max_results:
            break
    return results

def load_model(model_id, is_pretrained):
    # if model_id.startswith('timm/'):
    #     # timmを使用してモデルをロード
    #     model = timm.create_model(model_id.split('/')[-1], pretrained=is_pretrained)
    #     return model, None  # VGG16にはtokenizerが不要なのでNoneを返す
    # else:
    #     # 他のモデルの場合は既存のコードを使用
    #     if is_pretrained:
    #         model = AutoModel.from_pretrained(model_id)
    #         tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     else:
    #         config = AutoConfig.from_pretrained(model_id)
    #         model = AutoModel.from_config(config)
    #         tokenizer = AutoTokenizer.from_pretrained(model_id)
    #     return model, tokenizer
    
    if is_pretrained:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            model_max_length=512
        )
    else:
        config = AutoConfig.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_config(config)
        tokenizer = AutoTokenizer.from_config(config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def prepare_model(query, is_pretrained=True):
    results = search_models(query)
    selected_model = results[0]
    model, tokenizer = load_model(selected_model, is_pretrained)
    return model, tokenizer

if __name__ == "__main__":
    query = input("検索キーワードを入力: ")
    model, tokenizer = prepare_model(query)
    print(model, tokenizer)