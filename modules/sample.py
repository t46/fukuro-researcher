from vllm import LLM, SamplingParams


model_path = "meta-llama/Llama-2-7b-chat-hf"
temperature = 1.0
top_p = 1.0
max_tokens = 128
quantization = None
    
llm = LLM(
    model=model_path,
    quantization=quantization,
    dtype="bfloat16",
)

sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    max_tokens=max_tokens
)

prompts = ["こんにちは", "おはようございます", "おやすみなさい"]

responses = llm.generate(prompts, sampling_params=sampling_params)

for prompt, response in zip(prompts, responses):
    print("prompt:", prompt)
    print("output:", response.outputs[0].text.strip())
    print("logprob:", response.outputs[0].cumulative_logprob)