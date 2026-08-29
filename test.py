from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',  # Ollama 的 OpenAI 兼容端点
    api_key='ollama'  # 必填但不会验证，随便填即可
)

response = client.chat.completions.create(
    model='qwen3',
    messages=[{'role': 'user', 'content': '你好'}]
)
print(response.choices[0].message.content)