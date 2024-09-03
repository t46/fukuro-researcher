import ollama

def run_llm(model_name: str, message: str) -> str:
    response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': message,
                },
                ])
    return response.content