import ollama
import os
import requests
import json
from anthropic import Anthropic

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def run_llm(model_name: str, message: str, format: str = None) -> str:
    if model_name.startswith("claude"):
        anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

        # メッセージの送信
        message = anthropic.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": message
            }]
        )

        # レスポンスの取得
        response = message.content[0].text
        print(response)
        return response
    
    else:
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

# def run_llm(model_name: str, message: str):

#     response = requests.post(
#     url="https://openrouter.ai/api/v1/chat/completions",
#     headers={
#         "Authorization": f"Bearer {OPENROUTER_API_KEY}"
#     },
#     data=json.dumps({
#         "model": model_name,
#         "messages": [
#         {
#                 "role": "user",
#                 "content": message
#             }
#         ]
        
#     })
#     )
#     print(response.json())
#     return response.json()['choices'][0]['message']['content']



# string から <..> </..> で囲まれた部分を取り出す。例えば <query> ... </query> で囲まれた部分を取り出す。
def extract_content_between_tags(string: str, start_tag: str, end_tag: str) -> str:
    start_index = string.find(start_tag)
    end_index = string.find(end_tag, start_index)
    return string[start_index + len(start_tag):end_index]