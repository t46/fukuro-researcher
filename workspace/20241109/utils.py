import ollama
import os
from anthropic import Anthropic

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


def run_llm(model_name: str, message: str, format: str = None) -> str:
    if model_name.startswith("claude"):
        anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

        message = anthropic.messages.create(
            model=model_name,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": message
            }]
        )

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


def extract_content_between_tags(string: str, start_tag: str, end_tag: str) -> str:
    start_index = string.find(start_tag)
    end_index = string.find(end_tag, start_index)
    return string[start_index + len(start_tag):end_index]