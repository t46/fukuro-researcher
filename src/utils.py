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

# string から <..> </..> で囲まれた部分を取り出す。例えば <query> ... </query> で囲まれた部分を取り出す。
def extract_content_between_tags(string: str, start_tag: str, end_tag: str) -> str:
    start_index = string.find(start_tag)
    end_index = string.find(end_tag, start_index)
    return string[start_index + len(start_tag):end_index]