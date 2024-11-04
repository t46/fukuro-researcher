import ollama

def run_llm(model_name: str, message: str, format: str = None) -> str:
    args = {
        'model': model_name,
        'messages': [
            {
                'role': 'user',
                'content': message,
            },
        ],
    }
    if format == "json":
        args['format'] = 'json'
    response = ollama.chat(**args)
    print(response['message']['content'])
    return response['message']['content']